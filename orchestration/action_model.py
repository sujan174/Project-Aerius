"""
Core action model: Represents a single verifiable action.

This module defines the data structures for actions that flow through
the confirmation system.

Classes:
- ActionType: Enum for types of actions (create, update, delete, send, etc.)
- RiskLevel: Enum for risk assessment
- ActionStatus: Enum for action lifecycle states
- FieldConstraint: Validation constraints for action parameters
- FieldInfo: Metadata about an editable field
- Action: Complete action representation with audit trail
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import uuid


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
    LOW = "low"           # No confirmation needed
    MEDIUM = "medium"     # Ask before proceeding
    HIGH = "high"         # Ask, allow edits, require explicit yes


class ActionStatus(str, Enum):
    """Lifecycle status of an action"""
    PENDING = "pending"           # Waiting for user confirmation
    CONFIRMED = "confirmed"       # User approved
    REJECTED = "rejected"         # User denied
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
                import re
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
    Represents a single action that might need user confirmation.
    Complete lifecycle from creation through execution.
    """

    # Identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

    # Action Classification
    agent_name: str = ""            # Which agent performs this
    action_type: ActionType = ActionType.EXECUTE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    reason_for_confirmation: str = ""   # Why we're asking

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
    batch_id: Optional[str] = None  # Which confirmation batch was this in?

    def apply_edits(self) -> Dict[str, Any]:
        """
        Merge user edits with original parameters.
        User edits override original values.
        """
        merged = self.parameters.copy()
        merged.update(self.user_edits)
        return merged

    def mark_confirmed(self) -> None:
        """User confirmed this action"""
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
