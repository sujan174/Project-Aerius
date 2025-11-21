#!/usr/bin/env python3
"""
AI Workspace Orchestrator - Main Entry Point

Beautiful terminal interface powered by Rich.
Claude Code-quality user experience.

Usage:
    python main.py              # Start with enhanced UI
    python main.py --verbose    # Show debug information
    python main.py --simple     # Use simple UI (fallback)

Author: AI System
Version: 3.0
"""

import asyncio
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import OrchestratorAgent
from ui.enhanced_terminal_ui import EnhancedTerminalUI


# ============================================================================
# ASYNC GENERATOR CLEANUP FIX
# ============================================================================
#
# MCP's stdio_client uses anyio task groups which require __aexit__() to be
# called from the same task as __aenter__(). During application shutdown,
# Python's garbage collector may finalize async generators from a different
# task context, causing RuntimeError: "Attempted to exit cancel scope in a
# different task than it was entered in"
#
# This fix installs a custom async generator finalizer that suppresses these
# specific errors during shutdown, which don't affect functionality.
# ============================================================================

_shutting_down = False

def _create_safe_finalizer(loop):
    """Create a finalizer that safely closes async generators during shutdown"""
    def safe_finalizer(agen):
        if _shutting_down:
            # During shutdown, suppress task context errors from MCP cleanup
            try:
                loop.run_until_complete(agen.aclose())
            except RuntimeError as e:
                # Suppress "cancel scope in different task" errors
                if "cancel scope" in str(e) or "different task" in str(e):
                    pass
                else:
                    raise
            except Exception:
                # Suppress all errors during shutdown - connections will close anyway
                pass
        else:
            # Normal operation - use default behavior
            loop.run_until_complete(agen.aclose())
    return safe_finalizer

def setup_asyncgen_hooks():
    """Set up async generator hooks to handle MCP cleanup gracefully"""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            sys.set_asyncgen_hooks(finalizer=_create_safe_finalizer(loop))
    except RuntimeError:
        # No event loop available yet, will be set up later
        pass

# Suppress specific warnings from anyio/MCP during cleanup
warnings.filterwarnings(
    "ignore",
    message=".*cancel scope.*different task.*",
    category=RuntimeWarning
)

# Custom unraisable hook to suppress MCP cleanup errors during shutdown
_original_unraisable_hook = sys.unraisablehook

def _custom_unraisable_hook(unraisable):
    """Suppress MCP-related errors during shutdown"""
    if _shutting_down:
        # Check if this is an MCP/anyio cleanup error
        err_str = str(unraisable.exc_value) if unraisable.exc_value else ""
        obj_str = str(unraisable.object) if unraisable.object else ""

        # Suppress cancel scope and stdio_client errors during shutdown
        if any(pattern in err_str or pattern in obj_str for pattern in [
            "cancel scope", "different task", "stdio_client", "TaskGroup"
        ]):
            return  # Suppress the error

    # Call original hook for other errors
    _original_unraisable_hook(unraisable)

sys.unraisablehook = _custom_unraisable_hook


async def main():
    """Main entry point with enhanced UI"""
    global _shutting_down

    # Parse arguments
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    simple_ui = "--simple" in sys.argv

    if simple_ui:
        # Use original simple UI
        orchestrator = OrchestratorAgent(
            connectors_dir="connectors",
            verbose=verbose
        )
        try:
            await orchestrator.run_interactive()
        finally:
            _shutting_down = True
        return

    # Use enhanced UI
    ui = EnhancedTerminalUI(verbose=verbose)

    try:
        # Initialize orchestrator
        orchestrator = OrchestratorAgent(
            connectors_dir="connectors",
            verbose=False  # UI handles display
        )

        # Show header
        ui.print_header(orchestrator.session_id)

        # Discover agents with beautiful display
        ui.console.print("[bold cyan]🔌 Discovering Agents...[/bold cyan]\n")

        # Load agents (using existing method)
        await orchestrator.discover_and_load_agents()

        # Show agent summary
        loaded_agents = []
        failed_agents = []

        for agent_name, health in orchestrator.agent_health.items():
            if health['status'] == 'healthy':
                caps = orchestrator.agent_capabilities.get(agent_name, [])
                loaded_agents.append({
                    'name': agent_name,
                    'capabilities': caps[:3],  # Show first 3
                    'total': len(caps)
                })
            else:
                failed_agents.append(agent_name)

        # Print table of agents
        if loaded_agents:
            from rich.table import Table
            from rich import box

            table = Table(
                show_header=True,
                header_style="bold cyan",
                border_style="dim",
                box=box.ROUNDED
            )
            table.add_column("Agent", style="white")
            table.add_column("Capabilities", style="dim")

            for agent in loaded_agents:
                caps_text = ", ".join(agent['capabilities'])
                if agent['total'] > 3:
                    caps_text += f" (+{agent['total']-3} more)"
                table.add_row(agent['name'].title(), caps_text)

            ui.console.print(table)
            ui.console.print()

        ui.print_loaded_summary(len(loaded_agents), len(failed_agents))

        # Start interactive session
        await run_interactive_session(orchestrator, ui)

    except KeyboardInterrupt:
        ui.print_goodbye()
    except Exception as e:
        ui.print_error(str(e))
        if verbose:
            import traceback
            ui.print_error(str(e), traceback.format_exc())
    finally:
        # Signal shutdown to suppress MCP cleanup errors
        _shutting_down = True
        # Cleanup
        if 'orchestrator' in locals():
            await orchestrator.cleanup()


async def run_interactive_session(orchestrator: OrchestratorAgent, ui: EnhancedTerminalUI):
    """Run interactive chat session with enhanced UI"""

    message_count = 0

    while True:
        try:
            # Show prompt
            ui.print_prompt()

            # Get user input
            user_input = input().strip()

            # Handle exit
            if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                # Show stats
                import time
                duration = time.time() - orchestrator.analytics.start_time if hasattr(orchestrator, 'analytics') else 0
                stats = {
                    'duration': f"{int(duration)}s" if duration else "N/A",
                    'message_count': message_count,
                    'agent_calls': sum(ui.agent_calls.values()),
                    'success_rate': "N/A"
                }
                ui.print_session_stats(stats)
                ui.print_goodbye()
                break

            if not user_input:
                continue

            message_count += 1

            # Show thinking
            ui.print_thinking()

            # Process message with orchestrator
            response = await process_with_ui(orchestrator, user_input, ui)

            # Show response
            ui.print_response(response)

        except KeyboardInterrupt:
            ui.print_goodbye()
            break
        except Exception as e:
            ui.print_error(str(e))
            if ui.verbose:
                import traceback
                ui.print_error(str(e), traceback.format_exc())


async def process_with_ui(
    orchestrator: OrchestratorAgent,
    user_message: str,
    ui: EnhancedTerminalUI
) -> str:
    """Process message and update UI with tool calls"""

    # Hook into orchestrator's tool calling
    original_call_sub_agent = orchestrator.call_sub_agent

    async def wrapped_call_sub_agent(agent_name: str, instruction: str, context: dict = None):
        """Wrapped version that shows tool calls in UI"""

        # Show tool call
        ui.print_tool_call(agent_name, "processing...")

        try:
            result = await original_call_sub_agent(agent_name, instruction, context)

            # Show success
            success = not (result.startswith("Error") or result.startswith("⚠️"))
            ui.print_tool_result(success, result[:50] if success else result[:100])

            return result

        except Exception as e:
            ui.print_tool_result(False, str(e)[:100])
            raise

    # Temporarily replace method
    orchestrator.call_sub_agent = wrapped_call_sub_agent

    try:
        # Process message
        response = await orchestrator.process_message(user_message)
        return response
    finally:
        # Restore original
        orchestrator.call_sub_agent = original_call_sub_agent


if __name__ == "__main__":
    asyncio.run(main())
