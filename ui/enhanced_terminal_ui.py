"""
Enhanced Terminal UI - Claude Code Quality

Beautiful, interactive terminal interface with:
- Markdown rendering
- Syntax highlighting
- Streaming responses
- Status panels
- Progress indicators
- Clean visual design

Author: AI System
Version: 3.0 (Claude Code Quality)
"""

import sys
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from rich.columns import Columns
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich library for better UI...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich.rule import Rule
    from rich.columns import Columns
    from rich import box


class EnhancedTerminalUI:
    """
    Claude Code-quality terminal interface.

    Features:
    - Beautiful markdown rendering
    - Syntax-highlighted code blocks
    - Streaming responses
    - Agent status display
    - Progress indicators
    - Clean, professional design
    """

    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose

        # Theme colors
        self.colors = {
            'primary': '#00D9FF',      # Cyan
            'success': '#00FF88',      # Green
            'warning': '#FFB800',      # Orange
            'error': '#FF4444',        # Red
            'info': '#8B8BFF',         # Blue
            'muted': '#888888',        # Gray
            'agent': '#FF88FF',        # Purple
            'user': '#FFFFFF',         # White
        }

        # Session state
        self.session_start = datetime.now()
        self.message_count = 0
        self.agent_calls = {}

        # Setup prompt_toolkit for better input
        self._setup_input()

    def _setup_input(self):
        """Setup prompt_toolkit for multi-line input with Alt+Enter"""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys

        # Setup key bindings: Enter to submit, Alt+Enter for newline
        kb = KeyBindings()

        @kb.add(Keys.Enter)
        def _(event):
            """Enter submits"""
            event.current_buffer.validate_and_handle()

        @kb.add(Keys.Escape, Keys.Enter)  # Alt+Enter
        def _(event):
            """Alt+Enter for newline"""
            event.current_buffer.insert_text('\n')

        self.prompt_session = PromptSession(
            multiline=True,
            enable_history_search=True,
            key_bindings=kb
        )

    async def get_input(self) -> str:
        """Get user input with multi-line support (Alt+Enter for newline)"""
        try:
            from prompt_toolkit.formatted_text import HTML
            user_input = await self.prompt_session.prompt_async(
                HTML('<cyan>You › </cyan>'),
                vi_mode=False,
                mouse_support=False
            )
            return user_input.strip()
        except (KeyboardInterrupt, EOFError):
            return "exit"

    def clear_screen(self):
        """Clear terminal screen"""
        self.console.clear()

    def print_header(self, session_id: str):
        """Print beautiful header on startup"""
        self.clear_screen()

        # Create header panel
        header_text = Text()
        header_text.append("╔═══════════════════════════════════════════════════════════╗\n", style="bold cyan")
        header_text.append("║                                                           ║\n", style="bold cyan")
        header_text.append("║        ", style="bold cyan")
        header_text.append("AI WORKSPACE ORCHESTRATOR", style="bold white")
        header_text.append("                  ║\n", style="bold cyan")
        header_text.append("║                                                           ║\n", style="bold cyan")
        header_text.append("║           Your intelligent multi-agent assistant          ║\n", style="cyan")
        header_text.append("║                                                           ║\n", style="bold cyan")
        header_text.append("╚═══════════════════════════════════════════════════════════╝", style="bold cyan")

        self.console.print(header_text)
        self.console.print()

        # Session info
        info_panel = Panel(
            f"[cyan]Session:[/cyan] [white]{session_id[:8]}...[/white]\n"
            f"[cyan]Started:[/cyan] [white]{self.session_start.strftime('%I:%M %p')}[/white]\n"
            f"[cyan]Type 'exit' to quit[/cyan]",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 2)
        )
        self.console.print(info_panel)
        self.console.print()

    def print_agent_discovery(self, agents: List[Dict[str, Any]]):
        """Show agent discovery progress"""
        self.console.print()
        self.console.print("[bold cyan]Discovering Agents...[/bold cyan]")
        self.console.print()

        # Create table
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1)
        )

        table.add_column("Agent", style="white", width=20)
        table.add_column("Status", width=15)
        table.add_column("Capabilities", style="dim", width=40)

        for agent in agents:
            name = agent['name']
            status = agent['status']
            caps = agent.get('capabilities', [])

            if status == 'loaded':
                status_text = "[green]Loaded[/green]"
            elif status == 'failed':
                status_text = "[yellow]Skipped[/yellow]"
            else:
                status_text = "[dim]...[/dim]"

            caps_text = ", ".join(caps[:3])
            if len(caps) > 3:
                caps_text += f" (+{len(caps)-3} more)"

            table.add_row(name.title(), status_text, caps_text)

        self.console.print(table)
        self.console.print()

    def print_loaded_summary(self, loaded_count: int, failed_count: int):
        """Print summary after agent loading"""
        if failed_count > 0:
            summary = (
                f"[green]Loaded [bold white]{loaded_count}[/bold white] agent(s)  "
                f"[yellow]Skipped [bold white]{failed_count}[/bold white][/yellow]"
            )
        else:
            summary = f"[green]All [bold white]{loaded_count}[/bold white] agents loaded successfully[/green]"

        panel = Panel(
            summary,
            border_style="green",
            box=box.HEAVY,
            padding=(0, 2)
        )
        self.console.print(panel)
        self.console.print()

    def print_prompt(self):
        """Print user input prompt"""
        prompt_text = Text()
        prompt_text.append("┃ ", style="bold cyan")
        prompt_text.append("You", style="bold white")
        prompt_text.append(" › ", style="dim")

        self.console.print(prompt_text, end="")

    def print_user_message(self, message: str):
        """Echo user message with formatting"""
        # Don't echo since we already have prompt
        pass

    def print_thinking(self):
        """Show thinking indicator"""
        thinking = Text()
        thinking.append("┃ ", style="bold cyan")
        thinking.append("Assistant", style="bold #00D9FF")
        thinking.append(" is thinking...", style="dim italic")
        self.console.print(thinking)

    def print_assistant_header(self):
        """Print assistant response header"""
        header = Text()
        header.append("\n┃ ", style="bold cyan")
        header.append("Assistant", style="bold #00D9FF")
        self.console.print(header)
        self.console.print("┃", style="bold cyan")

    def print_response(self, response: str):
        """Print assistant response with markdown rendering"""
        # Parse and render markdown
        self.print_assistant_header()

        # Simple approach: just render the whole response as markdown
        md = Markdown(response)

        # Create a panel for the response
        response_panel = Panel(
            md,
            border_style="dim cyan",
            box=box.ROUNDED,
            padding=(1, 2),
            title="[dim]Response[/dim]",
            title_align="left"
        )

        self.console.print(response_panel)
        self.console.print()

    def print_streaming_response(self, response_generator):
        """Stream response as it's generated (future enhancement)"""
        # For now, just collect and print
        full_response = ""
        for chunk in response_generator:
            full_response += chunk

        self.print_response(full_response)

    def print_tool_call(self, agent_name: str, tool_name: str):
        """Show when a tool is being called"""
        # Track agent calls
        self.agent_calls[agent_name] = self.agent_calls.get(agent_name, 0) + 1

        status = Text()
        status.append("┃  ", style="bold cyan")
        status.append(f"Calling ", style="dim")
        status.append(agent_name.title(), style="bold #FF88FF")
        status.append(f" → {tool_name}", style="dim")

        self.console.print(status)

    def print_tool_result(self, success: bool, message: Optional[str] = None):
        """Show tool result"""
        if success:
            status = Text()
            status.append("┃  ", style="bold cyan")
            status.append("Success", style="dim green")
            if message:
                status.append(f" • {message[:50]}", style="dim")
            self.console.print(status)
        else:
            status = Text()
            status.append("┃  ", style="bold cyan")
            status.append("Failed", style="dim red")
            if message:
                status.append(f" • {message[:50]}", style="dim")
            self.console.print(status)

    def print_error(self, error: str, traceback_str: Optional[str] = None):
        """Print formatted error"""
        self.console.print()

        error_panel = Panel(
            f"[bold red]Error[/bold red]\n\n{error}",
            title="[red]Error[/red]",
            border_style="red",
            box=box.HEAVY,
            padding=(1, 2)
        )
        self.console.print(error_panel)

        if traceback_str and self.verbose:
            self.console.print("\n[dim]Traceback:[/dim]")
            syntax = Syntax(
                traceback_str,
                "python",
                theme="monokai",
                line_numbers=False
            )
            self.console.print(syntax)

        self.console.print()


    def print_goodbye(self):
        """Print goodbye message"""
        self.console.print()

        goodbye_text = Text()
        goodbye_text.append("┃ ", style="bold cyan")
        goodbye_text.append("Goodbye! ", style="bold white")
        goodbye_text.append("Thanks for using the orchestrator.", style="dim")

        self.console.print(goodbye_text)
        self.console.print()

    def show_progress(self, description: str, total: Optional[int] = None):
        """Show progress bar for long operations"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def print_divider(self):
        """Print subtle divider"""
        self.console.print(Rule(style="dim"))

    # Helper methods

    def _split_markdown(self, text: str) -> List[tuple]:
        """Split markdown into text and code blocks"""
        parts = []
        current = ""
        in_code = False
        code_block = ""

        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('```'):
                if in_code:
                    # End code block
                    parts.append(('code', code_block))
                    code_block = ""
                    in_code = False
                else:
                    # Start code block
                    if current:
                        parts.append(('text', current))
                        current = ""
                    in_code = True
                    code_block = line + '\n'
            else:
                if in_code:
                    code_block += line + '\n'
                else:
                    current += line + '\n'

        # Add remaining
        if current:
            parts.append(('text', current))
        if code_block:
            parts.append(('code', code_block))

        return parts

    def _parse_code_block(self, block: str) -> tuple:
        """Extract language and code from code block"""
        lines = block.strip().split('\n')
        if lines[0].startswith('```'):
            lang = lines[0][3:].strip() or 'text'
            code = '\n'.join(lines[1:])
            if code.endswith('```'):
                code = code[:-3].strip()
            return lang, code
        return 'text', block


# Convenience instance
enhanced_ui = EnhancedTerminalUI()
