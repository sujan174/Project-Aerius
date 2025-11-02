"""
Orchestration system: Manages action verification, confirmation, and execution.

This package handles the verification workflow that sits between the LLM's
function calls and actual agent execution.

Main Components:
- action_model: Core Action dataclass and related types
- confirmation_queue: Batch queue management
- action_parser: Parse instructions into structured Actions
- action_enricher: Fetch context and details for actions before confirmation
"""

from .action_model import (
    Action,
    ActionType,
    ActionStatus,
    RiskLevel,
    FieldInfo,
    FieldConstraint,
)
from .confirmation_queue import ConfirmationQueue, ConfirmationBatch
from .action_enricher import ActionEnricher

__all__ = [
    "Action",
    "ActionType",
    "ActionStatus",
    "RiskLevel",
    "FieldInfo",
    "FieldConstraint",
    "ConfirmationQueue",
    "ConfirmationBatch",
    "ActionEnricher",
]
