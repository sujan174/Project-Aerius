"""
Orchestration system: Manages action verification and execution.

This package handles the verification workflow that sits between the LLM's
function calls and actual agent execution.

Architecture:
- actions.py: Action models, parsing, and enrichment

Complete action lifecycle from instruction through execution.

Author: AI System
Version: 4.0 - Consolidated structure
"""

# Action management (from actions.py)
from .actions import (
    Action, ActionType, RiskLevel, ActionStatus,
    FieldInfo, FieldConstraint,
    ActionParser, ActionEnricher
)

__all__ = [
    # Action models
    "Action",
    "ActionType",
    "ActionStatus",
    "RiskLevel",
    "FieldInfo",
    "FieldConstraint",

    # Action processing
    "ActionParser",
    "ActionEnricher",
]

__version__ = "4.0.0"
