"""
Action enricher: Fetches context and details for actions before confirmation.

This module enhances actions with real data from agents so users can see
exactly what will be affected before approving.

Examples:
- DELETE action: Show which Jira issues will be deleted (with titles)
- CREATE action: Show what will be created
- SEND action: Show who will receive the message
"""

from typing import Any, Dict, List, Optional
import re
from orchestration.action_model import Action, ActionType


class ActionEnricher:
    """
    Enriches actions with context data from agents.
    Fetches details that users need to see before approving.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def enrich_action(self, action: Action, agent: Optional[Any] = None) -> None:
        """
        Enhance an action with contextual details.
        Calls agent to fetch data if needed.

        Args:
            action: Action to enrich (modified in-place)
            agent: Agent instance (optional, for fetching context)
        """

        # Route to agent-specific enrichment
        if action.agent_name == 'jira':
            await self._enrich_jira_action(action, agent)
        elif action.agent_name == 'slack':
            await self._enrich_slack_action(action, agent)
        elif action.agent_name == 'github':
            await self._enrich_github_action(action, agent)
        else:
            # Generic enrichment
            await self._enrich_generic_action(action, agent)

    async def _enrich_jira_action(self, action: Action, agent: Optional[Any] = None) -> None:
        """Enrich Jira actions with issue details"""

        if action.action_type == ActionType.DELETE:
            # Extract issue keys from instruction
            issue_keys = self._extract_jira_keys(action.instruction)
            if issue_keys:
                action.details = {
                    'issue_keys': issue_keys,
                    'count': len(issue_keys),
                    'description': f"Will permanently delete {len(issue_keys)} Jira issue(s)"
                }

                # Try to fetch full issue details if agent is available
                if agent and hasattr(agent, 'execute'):
                    try:
                        fetch_instruction = f"Get details for issues: {', '.join(issue_keys)}"
                        details_response = await agent.execute(fetch_instruction)
                        action.details['full_details'] = details_response
                    except Exception as e:
                        if self.verbose:
                            print(f"[ENRICHER] Could not fetch Jira details: {e}")

        elif action.action_type == ActionType.CREATE:
            action.details = {
                'description': "Will create a new Jira issue",
                'summary': action.parameters.get('title', 'Untitled'),
                'project': action.parameters.get('project', 'Unknown')
            }

        elif action.action_type == ActionType.UPDATE:
            issue_key = self._extract_jira_keys(action.instruction)
            action.details = {
                'description': f"Will update Jira issue(s): {', '.join(issue_key) if issue_key else 'Unknown'}",
                'changes': action.parameters
            }

    async def _enrich_slack_action(self, action: Action, agent: Optional[Any] = None) -> None:
        """Enrich Slack actions with channel/recipient info"""

        if action.action_type == ActionType.SEND:
            # Extract channel/recipient from instruction
            channel_match = re.search(r'(?:to|in|channel)[:\s]+([#@\w\-]+)', action.instruction, re.IGNORECASE)
            channel = channel_match.group(1) if channel_match else 'Unknown'

            # Extract message
            message_match = re.search(r'(?:message|text)[:\s]*["\']?([^"\']+)', action.instruction, re.IGNORECASE)
            message = message_match.group(1) if message_match else action.parameters.get('message', '')

            action.details = {
                'channel': channel,
                'message_preview': message[:100] + ('...' if len(message) > 100 else ''),
                'full_message': message,
                'description': f"Will send message to {channel}"
            }

    async def _enrich_github_action(self, action: Action, agent: Optional[Any] = None) -> None:
        """Enrich GitHub actions with PR/issue info"""

        if action.action_type == ActionType.SEND:
            # Likely a comment on a PR/issue
            action.details = {
                'type': 'comment',
                'description': 'Will post a comment',
                'preview': action.parameters.get('comment', '')[:100]
            }

        elif action.action_type == ActionType.DELETE:
            action.details = {
                'description': 'Will delete GitHub resource',
                'target': action.parameters.get('target', 'Unknown')
            }

    async def _enrich_generic_action(self, action: Action, agent: Optional[Any] = None) -> None:
        """Generic enrichment for unknown agents"""
        action.details = {
            'description': f"{action.action_type.value.upper()} operation on {action.agent_name}",
            'parameters': action.parameters
        }

    def _extract_jira_keys(self, text: str) -> List[str]:
        """Extract Jira issue keys (e.g., KAN-123) from text"""
        pattern = r'\b([A-Z]+-\d+)\b'
        matches = re.findall(pattern, text)
        return matches

    def get_action_summary(self, action: Action) -> str:
        """Get a human-readable summary of what will happen"""

        if not hasattr(action, 'details') or not action.details:
            return action.reason_for_confirmation

        details = action.details
        return details.get('description', action.reason_for_confirmation)

    def get_action_context_lines(self, action: Action) -> List[str]:
        """Get detailed context lines for display"""

        lines = []

        if not hasattr(action, 'details') or not action.details:
            return lines

        details = action.details

        # Add description
        if 'description' in details:
            lines.append(f"  {details['description']}")

        # Add issue keys
        if 'issue_keys' in details:
            keys_str = ', '.join(details['issue_keys'])
            lines.append(f"  Issues: {keys_str}")

        # Add message preview
        if 'message_preview' in details:
            lines.append(f"  Message: {details['message_preview']}")

        # Add channel
        if 'channel' in details:
            lines.append(f"  Channel: {details['channel']}")

        # Add full details if available
        if 'full_details' in details:
            lines.append(f"\n  Details:")
            detail_text = details['full_details']
            if isinstance(detail_text, str):
                # Take first few lines
                detail_lines = detail_text.split('\n')[:3]
                for line in detail_lines:
                    if line.strip():
                        lines.append(f"    {line.strip()}")

        return lines
