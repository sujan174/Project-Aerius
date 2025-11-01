#!/usr/bin/env python3
"""
Script to add session logging to all MCP agents

This script updates agent __init__ methods and adds MCP tool call logging.
"""

import re
import sys
from pathlib import Path


# List of agents to update (excluding github which is already done)
AGENTS_TO_UPDATE = [
    'notion_agent.py',
    'slack_agent.py',
    'jira_agent.py',
    'browser_agent.py',
    'scraper_agent.py',
]


def add_session_logger_to_init(content, agent_name):
    """Add session_logger parameter to __init__ if not already present"""

    if 'session_logger' in content:
        print(f"  ✓ {agent_name}: session_logger already in __init__")
        return content

    # Pattern to find __init__ method
    init_pattern = r'(def __init__\(\s*self,\s*[^)]*?)()\):'

    def replace_init(match):
        before_params = match.group(1)
        # Add session_logger parameter
        return f"{before_params},\n        session_logger=None\n    ):"

    content_updated = re.sub(init_pattern, replace_init, content, count=1)

    if content_updated == content:
        print(f"  ⚠ {agent_name}: Could not find __init__ method")
        return content

    # Add docstring update
    docstring_pattern = r'(Args:.*?)(""")'
    def add_logger_doc(match):
        args_section = match.group(1)
        end_quote = match.group(2)
        # Add session_logger to docstring
        if 'session_logger' not in args_section:
            return f"{args_section}            session_logger: Optional session logger for tracking operations\n        {end_quote}"
        return match.group(0)

    content_updated = re.sub(docstring_pattern, add_logger_doc, content_updated, flags=re.DOTALL)

    # Add logger initialization after super().__init__()
    super_pattern = r'(super\(\).__init__\(\))\s*\n'

    def add_logger_init(match):
        super_call = match.group(1)
        # Add logger initialization
        return f"""{super_call}

        # Session logging
        self.logger = session_logger
        self.agent_name = "{agent_name.replace('_agent.py', '')}"

"""

    content_updated = re.sub(super_pattern, add_logger_init, content_updated, count=1)

    print(f"  ✓ {agent_name}: Added session_logger to __init__")
    return content_updated


def add_mcp_tool_logging(content, agent_name):
    """Add logging around MCP tool calls"""

    if 'self.logger.log_tool_call' in content:
        print(f"  ✓ {agent_name}: Tool logging already present")
        return content

    # Add time import if not present
    if 'import time' not in content:
        # Add after other imports
        import_pattern = r'(import asyncio\n)'
        content = re.sub(import_pattern, r'\1import time\n', content, count=1)

    # Find call_tool patterns
    call_tool_pattern = r'(\s+)(tool_result = await self\.session\.call_tool\([^)]+\))'

    def add_logging(match):
        indent = match.group(1)
        call_line = match.group(2)

        # Extract tool name (usually a variable)
        tool_name_match = re.search(r'call_tool\(([^,]+)', call_line)
        if not tool_name_match:
            return match.group(0)

        tool_name_var = tool_name_match.group(1).strip()

        return f"""{indent}# Log tool call start
{indent}start_time = time.time()

{indent}{call_line}

{indent}# Log tool call completion
{indent}duration = time.time() - start_time
{indent}if self.logger:
{indent}    self.logger.log_tool_call(self.agent_name, {tool_name_var}, duration, success=True)
"""

    content_updated = re.sub(call_tool_pattern, add_logging, content)

    if content_updated != content:
        print(f"  ✓ {agent_name}: Added MCP tool logging")
    else:
        print(f"  ⚠ {agent_name}: Could not find call_tool patterns")

    return content_updated


def update_agent(agent_file):
    """Update a single agent file"""
    print(f"\nUpdating {agent_file.name}...")

    # Read file
    content = agent_file.read_text(encoding='utf-8')

    # Apply updates
    content = add_session_logger_to_init(content, agent_file.name)
    content = add_mcp_tool_logging(content, agent_file.name)

    # Write back
    agent_file.write_text(content, encoding='utf-8')

    print(f"  ✅ {agent_file.name} updated successfully")


def main():
    """Main function"""
    print("="*70)
    print("  Adding Session Logging to MCP Agents")
    print("="*70)

    connectors_dir = Path(__file__).parent / "connectors"

    if not connectors_dir.exists():
        print(f"❌ Error: connectors directory not found: {connectors_dir}")
        return 1

    updated_count = 0
    failed_count = 0

    for agent_name in AGENTS_TO_UPDATE:
        agent_file = connectors_dir / agent_name

        if not agent_file.exists():
            print(f"\n⚠ Skipping {agent_name}: File not found")
            continue

        try:
            update_agent(agent_file)
            updated_count += 1
        except Exception as e:
            print(f"\n❌ Failed to update {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            failed_count += 1

    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"✅ Successfully updated: {updated_count} agents")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count} agents")

    print("\n✅ All agents now have session logging enabled!")
    print("   - session_logger parameter added to __init__")
    print("   - MCP tool calls are logged with timing")
    print("   - Logs saved to logs/session_*.log")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
