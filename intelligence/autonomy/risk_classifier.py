"""
Risk Classification for Confidence-Based Autonomy

Classifies operations by risk level to determine whether to auto-execute or confirm first.
"""

from typing import List, Tuple
from enum import Enum

# Import base types from parent
from ..base_types import Intent, IntentType


class RiskLevel(Enum):
    """Risk level for operations - used for confidence-based autonomy"""
    LOW = "low"          # Read-only operations - auto-execute
    MEDIUM = "medium"    # Write operations - confirm if confidence < threshold
    HIGH = "high"        # Destructive operations - always confirm


class OperationRiskClassifier:
    """
    Classifies operation risk for confidence-based autonomy.

    Rules:
    - READ, SEARCH, ANALYZE = LOW risk → auto-execute
    - CREATE, UPDATE, COORDINATE, WORKFLOW = MEDIUM risk → confirm if confidence < 0.75
    - DELETE = HIGH risk → always confirm
    """

    @staticmethod
    def classify_risk(intents: List[Intent]) -> RiskLevel:
        """Classify risk level based on primary intent"""
        if not intents:
            return RiskLevel.MEDIUM

        # Get highest confidence intent
        primary_intent = max(intents, key=lambda i: i.confidence)

        # Classify based on intent type
        if primary_intent.type == IntentType.DELETE:
            return RiskLevel.HIGH

        elif primary_intent.type in [IntentType.READ, IntentType.SEARCH, IntentType.ANALYZE]:
            return RiskLevel.LOW

        elif primary_intent.type in [IntentType.CREATE, IntentType.UPDATE,
                                      IntentType.COORDINATE, IntentType.WORKFLOW]:
            return RiskLevel.MEDIUM

        else:  # UNKNOWN or other
            return RiskLevel.MEDIUM

    @staticmethod
    def should_confirm(risk_level: RiskLevel, confidence: float) -> Tuple[bool, str]:
        """
        Determine if user confirmation is needed.

        Returns:
            (needs_confirmation, reason)
        """
        if risk_level == RiskLevel.HIGH:
            return (True, "Destructive operation requires confirmation")

        elif risk_level == RiskLevel.MEDIUM:
            if confidence < 0.75:
                return (True, f"Medium risk operation with moderate confidence ({confidence:.2f})")
            else:
                return (False, f"Medium risk operation with high confidence ({confidence:.2f})")

        else:  # LOW risk
            return (False, "Read-only operation - safe to auto-execute")
