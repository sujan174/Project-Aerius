"""
Action Management System

This module consolidates action-related functionality:
- Action Models: Data structures for actions and their lifecycle
- Action Parsing: Converts instructions into structured actions
- Action Enrichment: Fetches context and details for actions

Complete action pipeline from raw instructions through execution.

Merged from:
- action_model.py
- action_parser.py
- action_enricher.py

Author: AI System
Version: 4.0 - Consolidated action system
"""

import re
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

from config import Config
from core.logger import get_logger
from core.input_validator import InputValidator

logger = get_logger(__name__)


# ============================================================================
# ACTION MODELS
# ============================================================================

class ActionType(str, Enum):
    """Types of actions that can be performed"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SEND = "send"
    NOTIFY = "notify"
    ARCHIVE = "archive"
    EXECUTE = "execute"
    READ = "read"


class RiskLevel(str, Enum):
    """How much user verification is needed"""
    LOW = "low"           # Low risk operation
    MEDIUM = "medium"     # Medium risk operation
    HIGH = "high"         # High risk operation


class ActionStatus(str, Enum):
    """Lifecycle status of an action"""
    PENDING = "pending"           # Waiting for execution
    CONFIRMED = "confirmed"       # Ready to execute
    REJECTED = "rejected"         # Cancelled
    EXECUTING = "executing"       # Being executed by agent
    SUCCEEDED = "succeeded"       # Execution completed successfully
    FAILED = "failed"             # Execution failed


@dataclass
class FieldConstraint:
    """Constraints on what values are valid for a field"""
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None         # regex for validation
    allowed_values: Optional[List[Any]] = None
    forbidden_values: Optional[List[Any]] = None

    def validate(self, value: Any) -> tuple:
        """
        Validate a value against constraints.
        Returns: (is_valid, error_message)
        """
        if isinstance(value, str):
            if self.min_length and len(value) < self.min_length:
                return False, f"Must be at least {self.min_length} characters"
            if self.max_length and len(value) > self.max_length:
                return False, f"Must be at most {self.max_length} characters"
            if self.pattern:
                if not re.match(self.pattern, value):
                    return False, f"Must match pattern: {self.pattern}"

        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"Must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Must be at most {self.max_value}"

        if self.allowed_values and value not in self.allowed_values:
            return False, f"Must be one of: {self.allowed_values}"

        if self.forbidden_values and value in self.forbidden_values:
            return False, f"Cannot be: {self.forbidden_values}"

        return True, None


@dataclass
class FieldInfo:
    """Metadata about an action parameter that can be edited"""
    display_label: str              # "Issue Title" vs "title"
    description: str                # User-friendly explanation
    field_type: str                 # 'string', 'number', 'bool', 'select', 'text'
    current_value: Any              # Value extracted from instruction
    editable: bool = True           # Can user change this?
    required: bool = True           # Must have a value?
    constraints: FieldConstraint = field(default_factory=FieldConstraint)
    examples: Optional[List[str]] = None  # Examples for user


@dataclass
class Action:
    """
    Represents a single action that can be executed.
    Complete lifecycle from creation through execution.
    """

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Action Classification
    agent_name: str = ""            # Which agent performs this
    action_type: ActionType = ActionType.EXECUTE
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # Original Instruction
    instruction: str = ""           # Raw instruction from LLM
    context: Optional[Any] = None   # Any context passed to agent

    # Parameters & Editability
    parameters: Dict[str, Any] = field(default_factory=dict)      # Extracted params
    field_info: Dict[str, FieldInfo] = field(default_factory=dict)  # Metadata per field

    # User Interaction
    status: ActionStatus = ActionStatus.PENDING
    user_edits: Dict[str, Any] = field(default_factory=dict)      # Edits user made
    user_decision_at: Optional[datetime] = None
    user_decision_note: Optional[str] = None                       # If rejected, why?

    # Execution Tracking
    executed_at: Optional[datetime] = None
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None

    # Audit Trail
    batch_id: Optional[str] = None  # Which batch was this in?

    # Enrichment Details
    details: Dict[str, Any] = field(default_factory=dict)

    def apply_edits(self) -> Dict[str, Any]:
        """Merge user edits with original parameters"""
        merged = self.parameters.copy()
        merged.update(self.user_edits)
        return merged

    def mark_ready(self) -> None:
        """Mark action as ready to execute"""
        self.status = ActionStatus.CONFIRMED
        self.user_decision_at = datetime.now()

    def mark_rejected(self, reason: str = "") -> None:
        """User rejected this action"""
        self.status = ActionStatus.REJECTED
        self.user_decision_at = datetime.now()
        self.user_decision_note = reason

    def mark_executing(self) -> None:
        """Action is now being executed"""
        self.status = ActionStatus.EXECUTING

    def mark_succeeded(self, result: str) -> None:
        """Action executed successfully"""
        self.status = ActionStatus.SUCCEEDED
        self.executed_at = datetime.now()
        self.execution_result = result

    def mark_failed(self, error: str) -> None:
        """Action failed during execution"""
        self.status = ActionStatus.FAILED
        self.executed_at = datetime.now()
        self.execution_error = error

    def is_ready_for_execution(self) -> bool:
        """Can we execute this action now?"""
        return self.status == ActionStatus.CONFIRMED

    def can_edit(self, field_name: str) -> bool:
        """Is a specific field editable?"""
        if field_name not in self.field_info:
            return False
        return self.field_info[field_name].editable

    def validate_edits(self) -> tuple:
        """
        Validate all user edits against constraints.
        Returns: (all_valid, list_of_error_messages)
        """
        errors = []

        for field_name, new_value in self.user_edits.items():
            if field_name not in self.field_info:
                errors.append(f"Unknown field: {field_name}")
                continue

            field = self.field_info[field_name]

            # Check if field is editable
            if not field.editable:
                errors.append(f"Field '{field_name}' is not editable")
                continue

            # Validate constraints
            is_valid, error_msg = field.constraints.validate(new_value)
            if not is_valid:
                errors.append(f"{field_name}: {error_msg}")

        return len(errors) == 0, errors


# ============================================================================
# ACTION PARSING
# ============================================================================

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
        """
        action = Action(
            agent_name=agent_name,
            instruction=instruction,
            context=context
        )

        # Step 1: Determine action type and risk
        action.action_type = self._detect_action_type(instruction)
        action.risk_level = self._detect_risk_level(agent_name, instruction, action.action_type)

        # Step 2: Generic parameter extraction
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

    def _parse_generic_action(self, action: Action) -> None:
        """Generic parsing - extracts common patterns from instructions"""
        instruction = action.instruction

        # Try to find quoted strings
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
                    editable=True
                )

    async def _enrich_with_agent_schema(
        self,
        action: Action,
        agent_schema: Dict[str, Any]
    ) -> None:
        """Use agent's action schema to enhance field metadata"""
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


# ============================================================================
# ACTION ENRICHMENT
# ============================================================================

class ActionEnricher:
    """
    Enriches actions with context data from agents.
    Fetches details that users need to see before approving.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    async def enrich_action(self, action: Action, agent: Optional[Any] = None) -> None:
        """
        Enhance an action with contextual details (with timeout and validation).
        Calls agent to fetch data if needed.
        """
        try:
            # Validate instruction before enrichment
            is_valid, error = InputValidator.validate_instruction(action.instruction)
            if not is_valid:
                logger.warning(f"Invalid instruction: {error}")
                action.details = {'error': f'Invalid instruction: {error}'}
                return

            # Run enrichment with timeout
            try:
                await asyncio.wait_for(
                    self._enrich_action_impl(action, agent),
                    timeout=Config.ENRICHMENT_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(f"Enrichment timeout for {action.agent_name}")
                action.details = {'error': f'Enrichment timed out (>{Config.ENRICHMENT_TIMEOUT}s)'}
                if Config.REQUIRE_ENRICHMENT_FOR_HIGH_RISK and action.risk_level.value == 'high':
                    logger.error(f"Blocking HIGH-RISK action due to failed enrichment: {action.id}")

        except Exception as e:
            logger.error(f"Unexpected error during enrichment: {str(e)}", exc_info=True)
            action.details = {'error': f'Enrichment error: {str(e)}'}

    async def _enrich_action_impl(self, action: Action, agent: Optional[Any] = None) -> None:
        """Internal enrichment implementation"""
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
            # Extract channel/recipient from instruction (safe from regex injection)
            channel = InputValidator.extract_slack_channel_safe(action.instruction)
            if not channel:
                channel = 'Unknown'

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
        """Extract Jira issue keys (e.g., KAN-123) from text (safe from regex injection)"""
        return InputValidator.extract_jira_keys_safe(text)

    def get_action_summary(self, action: Action) -> str:
        """Get a human-readable summary of what will happen"""
        if not hasattr(action, 'details') or not action.details:
            return f"Action type: {action.action_type.value}"

        details = action.details
        return details.get('description', f"Action type: {action.action_type.value}")

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
