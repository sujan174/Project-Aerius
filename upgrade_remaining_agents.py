#!/usr/bin/env python3
"""
Automated Agent Intelligence Upgrade Script

This script automatically upgrades Slack, GitHub, and Notion agents with
intelligence features by applying the proven patterns from the Jira agent.

Usage:
    python upgrade_remaining_agents.py

What it does:
1. Adds intelligence imports
2. Updates __init__ to accept shared_context and knowledge_base
3. Adds intelligence components (memory, knowledge, proactive)
4. Enhances execute() method with intelligence features
5. Adds helper methods for conversation memory and cross-agent coordination

Author: AI System
Version: 1.0
"""

import os
import re
from pathlib import Path


def backup_file(filepath):
    """Create a backup of the file"""
    backup_path = f"{filepath}.backup"
    with open(filepath, 'r') as f:
        content = f.read()
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"✅ Backed up {filepath} → {backup_path}")


def add_intelligence_imports(content, agent_name):
    """Add intelligence module imports"""
    # Find the base_agent import line
    pattern = r'(from connectors\.base_agent import BaseAgent)'

    replacement = r'\1\nfrom connectors.agent_intelligence import (\n    ConversationMemory,\n    WorkspaceKnowledge,\n    SharedContext,\n    ProactiveAssistant\n)'

    if 'from connectors.agent_intelligence import' in content:
        print(f"⏭️  {agent_name}: Intelligence imports already present")
        return content

    content = re.sub(pattern, replacement, content)
    print(f"✅ {agent_name}: Added intelligence imports")
    return content


def update_init_method(content, agent_name):
    """Update __init__ method to include intelligence components"""

    # Pattern to find the __init__ method definition
    init_pattern = r'(def __init__\(self, verbose: bool = False\):)'

    # New definition with intelligence parameters
    new_init = '''def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None
    ):'''

    # Check if already updated
    if 'shared_context: Optional[SharedContext]' in content:
        print(f"⏭️  {agent_name}: __init__ already updated")
        return content

    content = content.replace(
        'def __init__(self, verbose: bool = False):',
        new_init
    )

    # Add intelligence components after stats
    stats_pattern = r'(self\.stats = OperationStats\(\))'

    intelligence_components = '''\1

        # Intelligence Components
        self.memory = ConversationMemory()
        self.knowledge = knowledge_base or WorkspaceKnowledge()
        self.shared_context = shared_context
        self.proactive = ProactiveAssistant('{}', verbose)'''.format(agent_name.lower())

    if 'self.memory = ConversationMemory()' not in content:
        content = re.sub(stats_pattern, intelligence_components, content)
        print(f"✅ {agent_name}: Updated __init__ method")
    else:
        print(f"⏭️  {agent_name}: Intelligence components already present")

    return content


def enhance_execute_method(content, agent_name):
    """Enhance execute() method with intelligence features"""

    # Check if already enhanced
    if '_resolve_references' in content and 'Step 1: Resolve ambiguous references' in content:
        print(f"⏭️  {agent_name}: execute() already enhanced")
        return content

    # Pattern: Find the start of execute method after initialization check
    pattern = r'(if not self\.initialized:[\s\S]*?try:)'

    enhancement = r'''\1
            # Step 1: Resolve ambiguous references using conversation memory
            resolved_instruction = self._resolve_references(instruction)

            if resolved_instruction != instruction and self.verbose:
                print(f"[{} AGENT] Resolved instruction: {{resolved_instruction}}")

            # Step 2: Check for resources from other agents
            context_from_other_agents = self._get_cross_agent_context()
            if context_from_other_agents and self.verbose:
                print(f"[{} AGENT] Found context from other agents")

            # Use resolved instruction for the rest
            instruction = resolved_instruction'''.format(agent_name.upper(), agent_name.upper())

    content = re.sub(pattern, enhancement, content, count=1)
    print(f"✅ {agent_name}: Enhanced execute() method")

    return content


def add_helper_methods(content, agent_name):
    """Add intelligence helper methods"""

    # Check if already added
    if '_resolve_references' in content:
        print(f"⏭️  {agent_name}: Helper methods already present")
        return content

    # Find insertion point (before _extract_function_call or similar method)
    pattern = r'(    def _extract_function_call\(self, response\))'

    # Resource pattern varies by agent
    if agent_name.lower() == 'slack':
        resource_pattern = r'(C[A-Z0-9]{10}|[0-9]{10}\.[0-9]{6})'
        resource_type = 'message'
        url_template = 'https://workspace.slack.com/archives/{channel}/p{ts}'
    elif agent_name.lower() == 'github':
        resource_pattern = r'#(\d+)'
        resource_type = 'issue'
        url_template = 'https://github.com/{owner}/{repo}/issues/{num}'
    elif agent_name.lower() == 'notion':
        resource_pattern = r'notion\.so/([a-f0-9]{32})'
        resource_type = 'page'
        url_template = '{url}'  # Use full URL from response
    else:
        resource_pattern = r'[A-Z]+-\d+'
        resource_type = 'resource'
        url_template = 'https://example.com/{id}'

    helper_methods = f'''    # ========================================================================
    # INTELLIGENCE HELPER METHODS
    # ========================================================================

    def _resolve_references(self, instruction: str) -> str:
        """Resolve ambiguous references using conversation memory"""
        ambiguous_terms = ['it', 'that', 'this', 'the issue', 'the ticket', 'the message', 'the page']

        for term in ambiguous_terms:
            if term in instruction.lower():
                reference = self.memory.resolve_reference(term)
                if reference:
                    instruction = instruction.replace(term, reference)
                    instruction = instruction.replace(term.capitalize(), reference)
                    if self.verbose:
                        print(f"[{agent_name.upper()} AGENT] Resolved '{{term}}' → {{reference}}")
                    break

        return instruction

    def _get_cross_agent_context(self) -> str:
        """Get context from other agents"""
        if not self.shared_context:
            return ""

        all_resources = self.shared_context.get_all_resources()
        if not all_resources:
            return ""

        context_parts = []
        for resource in all_resources:
            if resource['agent'] != '{agent_name.lower()}':
                context_parts.append(
                    f"{{resource['agent'].capitalize()}} {{resource['type']}}: {{resource['id']}} ({{resource['url']}})"
                )

        return "; ".join(context_parts) if context_parts else ""

    def _remember_created_resources(self, response: str, instruction: str):
        """Extract and remember created resources"""
        import re

        pattern = r'{resource_pattern}'
        matches = re.findall(pattern, response)

        if matches:
            resource_id = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
            operation_type = 'create' if 'creat' in instruction.lower() else 'update'

            self.memory.remember(operation_type, resource_id, {{'instruction': instruction[:100]}})

            if self.shared_context:
                # Build resource URL (simplified - adjust per agent)
                resource_url = f"{resource_type}://{{resource_id}}"

                self.shared_context.share_resource(
                    '{agent_name.lower()}',
                    '{resource_type}',
                    resource_id,
                    resource_url,
                    {{'created_by': '{agent_name.lower()}_agent'}}
                )

                if self.verbose:
                    print(f"[{agent_name.upper()} AGENT] Shared {{resource_id}} with other agents")

    def _infer_operation_type(self, instruction: str) -> str:
        """Infer operation type from instruction"""
        instruction_lower = instruction.lower()

        if 'create' in instruction_lower or 'new' in instruction_lower:
            return 'create'
        elif 'update' in instruction_lower or 'edit' in instruction_lower:
            return 'update'
        elif 'delete' in instruction_lower or 'remove' in instruction_lower:
            return 'delete'
        elif 'search' in instruction_lower or 'find' in instruction_lower or 'list' in instruction_lower:
            return 'search'
        else:
            return 'unknown'

    \1'''

    content = re.sub(pattern, helper_methods, content)
    print(f"✅ {agent_name}: Added helper methods")

    return content


def add_proactive_suggestions(content, agent_name):
    """Add proactive suggestions at the end of execute()"""

    if 'Suggested next steps' in content:
        print(f"⏭️  {agent_name}: Proactive suggestions already present")
        return content

    # Pattern: Find return final_response in execute method
    pattern = r'(final_response = response\.text[\s\S]*?)(return final_response)'

    suggestions = r'''\1
            # Remember resources and add proactive suggestions
            self._remember_created_resources(final_response, instruction)

            operation_type = self._infer_operation_type(instruction)
            suggestions = self.proactive.suggest_next_steps(operation_type, {})

            if suggestions:
                final_response += "\\n\\n**💡 Suggested next steps:**\\n" + "\\n".join(f"  • {{s}}" for s in suggestions)

            \2'''

    content = re.sub(pattern, suggestions, content, count=1)
    print(f"✅ {agent_name}: Added proactive suggestions")

    return content


def upgrade_agent(filepath, agent_name):
    """Upgrade a single agent file"""
    print(f"\n{'='*60}")
    print(f"Upgrading {agent_name} Agent")
    print(f"{'='*60}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False

    # Backup first
    backup_file(filepath)

    # Read content
    with open(filepath, 'r') as f:
        content = f.read()

    # Apply upgrades
    content = add_intelligence_imports(content, agent_name)
    content = update_init_method(content, agent_name)
    content = enhance_execute_method(content, agent_name)
    content = add_helper_methods(content, agent_name)
    content = add_proactive_suggestions(content, agent_name)

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"\n✅ {agent_name} Agent upgrade complete!")
    return True


def main():
    """Main upgrade process"""
    print("🚀 Starting Agent Intelligence Upgrade")
    print("=" * 60)

    # Get project root
    script_dir = Path(__file__).parent
    connectors_dir = script_dir / "connectors"

    # Agents to upgrade
    agents = [
        ('slack_agent.py', 'Slack'),
        ('github_agent.py', 'GitHub'),
        ('notion_agent.py', 'Notion'),
    ]

    success_count = 0

    for filename, agent_name in agents:
        filepath = connectors_dir / filename
        if upgrade_agent(str(filepath), agent_name):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Upgrade Complete!")
    print(f"✅ Successfully upgraded {success_count}/{len(agents)} agents")
    print(f"{'='*60}")

    print("\n📝 Next Steps:")
    print("1. Review the changes in each agent file")
    print("2. Update orchestrator.py to pass shared_context and knowledge_base")
    print("3. Test the intelligent features")
    print("\nSee IMPLEMENTATION_SUMMARY.md for details!")


if __name__ == "__main__":
    main()
