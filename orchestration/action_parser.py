"""
Action parser: Converts raw agent instructions into structured Actions.

This module extracts parameters and metadata from natural language instructions,
creating structured Action objects ready for confirmation.
"""

import re
from typing import Any, Dict, List, Optional
from orchestration.action_model import (
    Action, ActionType, RiskLevel, ActionStatus,
    FieldInfo, FieldConstraint
)


class ActionParser:
    """
    Converts agent instructions into structured Action objects.
    Uses regex patterns and agent metadata to understand parameters.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def parse_instruction(
        self,
        agent_name: str,
        instruction: str,
        agent: Optional[Any] = None,
        context: Optional[Any] = None
    ) -> Action:
        """
        Main parsing method.
        Returns a fully-formed Action with extracted parameters.

        Args:
            agent_name: Name of agent that will execute
            instruction: Raw instruction from LLM
            agent: Agent instance (optional, for schema)
            context: Context data passed to agent

        Returns:
            Action: Structured action ready for confirmation
        """

        action = Action(
            agent_name=agent_name,
            instruction=instruction,
            context=context
        )

        # Step 1: Determine action type and risk
        action.action_type = self._detect_action_type(instruction)
        action.risk_level = self._detect_risk_level(agent_name, instruction, action.action_type)

        # Step 2: Set confirmation reason
        action.reason_for_confirmation = self._get_confirmation_reason(
            agent_name, action.action_type, action.risk_level
        )

        # Step 3: Generic parameter extraction (can be enhanced per agent)
        self._parse_generic_action(action)

        # Step 4: Get agent schema if available
        if agent and hasattr(agent, 'get_action_schema'):
            try:
                schema = await agent.get_action_schema()
                await self._enrich_with_agent_schema(action, schema)
            except Exception as e:
                if self.verbose:
                    print(f"[PARSER] Could not get agent schema: {e}")

        return action

    def _detect_action_type(self, instruction: str) -> ActionType:
        """Determine what kind of action this is"""
        instruction_lower = instruction.lower()

        # Map keywords to action types
        type_map = {
            ActionType.CREATE: ['create', 'add', 'make', 'new', 'write', 'generate'],
            ActionType.UPDATE: ['update', 'modify', 'change', 'edit', 'set', 'revise'],
            ActionType.DELETE: ['delete', 'remove', 'destroy', 'drop', 'purge'],
            ActionType.SEND: ['send', 'post', 'share', 'broadcast', 'message', 'reply'],
            ActionType.NOTIFY: ['notify', 'alert', 'announce', 'inform'],
            ActionType.ARCHIVE: ['archive', 'close', 'close out'],
        }

        for action_type, keywords in type_map.items():
            if any(kw in instruction_lower for kw in keywords):
                return action_type

        return ActionType.EXECUTE  # Default

    def _detect_risk_level(
        self,
        agent_name: str,
        instruction: str,
        action_type: ActionType
    ) -> RiskLevel:
        """Determine risk level of action"""

        # Destructive actions are high-risk
        if action_type in [ActionType.DELETE, ActionType.ARCHIVE]:
            return RiskLevel.HIGH

        # Broadcast actions are high-risk
        if action_type == ActionType.SEND:
            if any(word in instruction.lower() for word in ['everyone', '@channel', '@here']):
                return RiskLevel.HIGH

        # Other writes are medium-risk
        if action_type in [ActionType.CREATE, ActionType.UPDATE]:
            return RiskLevel.MEDIUM

        # Read operations are low-risk
        return RiskLevel.LOW

    def _get_confirmation_reason(
        self,
        agent_name: str,
        action_type: ActionType,
        risk_level: RiskLevel
    ) -> str:
        """Generate user-friendly reason for confirmation"""

        if risk_level == RiskLevel.HIGH:
            if action_type == ActionType.DELETE:
                return "This will permanently delete items. Please review carefully."
            elif action_type == ActionType.SEND:
                return "This will notify many people. Please confirm message content."

        elif risk_level == RiskLevel.MEDIUM:
            if action_type == ActionType.CREATE:
                return "Creating new item. You can review and edit details before confirming."
            elif action_type == ActionType.UPDATE:
                return "Updating existing item. Please review changes."

        return "Please confirm this action."

    def _parse_generic_action(self, action: Action) -> None:
        """
        Generic parsing - extracts common patterns from instructions.
        More specific parsing can be added per agent type.
        """

        instruction = action.instruction

        # Try to find quoted strings (common parameter format)
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_values = re.findall(quoted_pattern, instruction)

        # Try to find key-value patterns
        kv_pattern = r'(\w+)[:\s]+["\']?([^"\';\n]+)["\']?'
        matches = re.finditer(kv_pattern, instruction)

        for match in matches:
            key, value = match.groups()
            key_lower = key.lower().strip()

            # Skip common words that aren't parameters
            skip_words = {'the', 'with', 'in', 'to', 'for', 'from', 'and', 'or', 'at', 'by'}
            if key_lower in skip_words:
                continue

            value = value.strip()

            # Add to parameters if not already there
            if key_lower not in action.parameters:
                action.parameters[key_lower] = value

                # Add basic field info
                action.field_info[key_lower] = FieldInfo(
                    display_label=key.replace('_', ' ').title(),
                    description=f"Parameter: {key}",
                    field_type='string',
                    current_value=value,
                    editable=True  # Default to editable unless proven otherwise
                )

    async def _enrich_with_agent_schema(
        self,
        action: Action,
        agent_schema: Dict[str, Any]
    ) -> None:
        """
        Use agent's action schema to enhance field metadata.
        Agent schema tells us which fields are editable, required, etc.
        """

        action_type_key = str(action.action_type.value)
        if action_type_key not in agent_schema:
            return

        schema_for_type = agent_schema[action_type_key]
        schema_params = schema_for_type.get('parameters', {})

        for param_name, param_schema in schema_params.items():
            if param_name in action.parameters:
                # Update editability and other constraints
                editable = param_schema.get('editable', True)

                # Create or update FieldInfo with schema info
                field_info = FieldInfo(
                    display_label=param_schema.get('display_label', param_name),
                    description=param_schema.get('description', ''),
                    field_type=param_schema.get('type', 'string'),
                    current_value=action.parameters[param_name],
                    editable=editable,
                    required=param_schema.get('required', True)
                )

                # Add constraints if provided
                constraints_schema = param_schema.get('constraints', {})
                if constraints_schema:
                    field_info.constraints = FieldConstraint(
                        min_length=constraints_schema.get('min_length'),
                        max_length=constraints_schema.get('max_length'),
                        min_value=constraints_schema.get('min_value'),
                        max_value=constraints_schema.get('max_value'),
                        pattern=constraints_schema.get('pattern'),
                        allowed_values=constraints_schema.get('allowed_values'),
                        forbidden_values=constraints_schema.get('forbidden_values')
                    )

                action.field_info[param_name] = field_info
