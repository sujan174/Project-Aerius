"""
Error Classification and Handling System

Provides intelligent error categorization to distinguish between:
- Transient errors (retry)
- Permanent errors (stop)
- Capability gaps (inform user)
- Permission issues (require user action)
- Rate limiting (backoff and retry)

This enables smarter recovery and better user messaging.

Author: AI System
Version: 1.0
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class ErrorCategory(str, Enum):
    """Categories of errors with different handling strategies"""
    TRANSIENT = "transient"          # Temporary - retry with backoff
    RATE_LIMIT = "rate_limit"        # API rate limited - retry with longer delay
    CAPABILITY = "capability"        # API doesn't support this - don't retry
    PERMISSION = "permission"        # Access denied - require user action
    VALIDATION = "validation"        # Invalid input - don't retry
    UNKNOWN = "unknown"              # Unknown - assume retryable but inform


@dataclass
class ErrorClassification:
    """Complete error classification with recovery suggestions"""
    category: ErrorCategory
    is_retryable: bool
    explanation: str  # What happened in simple terms
    technical_details: Optional[str] = None  # Full error message
    suggestions: List[str] = None  # What user can do
    retry_delay_seconds: int = 0  # Backoff delay for rate limits

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ErrorClassifier:
    """
    Intelligent error classification system.

    Analyzes error messages to determine:
    - Root cause category
    - Whether to retry
    - How to inform the user
    - What to suggest next
    """

    # Error patterns for each category
    CAPABILITY_PATTERNS = [
        'does not support',
        'cannot fetch',
        'not available',
        'api does not',
        'is not supported',
        'not implemented',
        'unsupported operation',
        'cannot provide',
        'unable to retrieve',
    ]

    PERMISSION_PATTERNS = [
        'permission denied',
        'forbidden',
        'unauthorized',
        '401',
        '403',
        'access denied',
        'private repository',
        'not found',
        '404',
        'insufficient permissions',
        'access token',
    ]

    RATE_LIMIT_PATTERNS = [
        'rate limit',
        'rate limited',
        'too many requests',
        'quota exceeded',
        '429',
        '503',
        'throttled',
        'back off',
    ]

    TRANSIENT_PATTERNS = [
        'timeout',
        'timed out',
        'connection',
        'network',
        'temporarily',
        '502',
        '504',
        'gateway',
        'temporary',
        'service unavailable',
    ]

    VALIDATION_PATTERNS = [
        'invalid input',
        'invalid parameter',
        'required field',
        'bad request',
        '400',
        'validation error',
        'malformed',
    ]

    @staticmethod
    def classify(error_msg: str, agent_name: Optional[str] = None) -> ErrorClassification:
        """
        Classify an error message and return handling strategy.

        Args:
            error_msg: The error message string
            agent_name: Optional agent name for context-specific classification

        Returns:
            ErrorClassification with category, retry decision, and suggestions
        """
        error_lower = error_msg.lower()

        # Check each category in priority order
        # (more specific categories first)

        # 1. Check for CAPABILITY errors (permanent - don't retry)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.CAPABILITY_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.CAPABILITY,
                is_retryable=False,
                explanation="The underlying API or agent does not support this operation",
                technical_details=error_msg,
                suggestions=[
                    "• Try a different approach or agent",
                    "• Check the agent's stated capabilities",
                    "• Look for an alternative tool that supports this operation",
                ]
            )

        # 2. Check for PERMISSION errors (permanent - don't retry)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.PERMISSION_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.PERMISSION,
                is_retryable=False,
                explanation="Access denied - check permissions and resource status",
                technical_details=error_msg,
                suggestions=[
                    "• Verify the resource exists and is accessible",
                    "• Check if the repository/resource is private",
                    "• Verify API token has required permissions",
                    "• Confirm you have access to this resource",
                ]
            )

        # 3. Check for VALIDATION errors (permanent - don't retry)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.VALIDATION_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.VALIDATION,
                is_retryable=False,
                explanation="Invalid input provided - request cannot be processed",
                technical_details=error_msg,
                suggestions=[
                    "• Check the instruction syntax",
                    "• Verify all required parameters are provided",
                    "• Ensure values are in the correct format",
                ]
            )

        # 4. Check for RATE_LIMIT errors (retryable with backoff)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.RATE_LIMIT_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.RATE_LIMIT,
                is_retryable=True,
                explanation="API rate limit reached - will retry with delay",
                technical_details=error_msg,
                suggestions=[
                    "• Operation will be retried automatically",
                    "• Consider spacing out large operations",
                ],
                retry_delay_seconds=10  # Wait 10 seconds before retry
            )

        # 5. Check for TRANSIENT errors (retryable)
        if ErrorClassifier._matches_patterns(error_lower, ErrorClassifier.TRANSIENT_PATTERNS):
            return ErrorClassification(
                category=ErrorCategory.TRANSIENT,
                is_retryable=True,
                explanation="Temporary network or service issue - will retry",
                technical_details=error_msg,
                suggestions=[
                    "• Operation will be retried automatically",
                    "• If it persists, the service may be down",
                ],
                retry_delay_seconds=2
            )

        # 6. Default to UNKNOWN (assume retryable but flag)
        return ErrorClassification(
            category=ErrorCategory.UNKNOWN,
            is_retryable=True,
            explanation="An unexpected error occurred",
            technical_details=error_msg,
            suggestions=[
                "• Check agent logs for more details",
                "• Verify the instruction format",
                "• Try with a simpler or more specific request",
            ]
        )

    @staticmethod
    def _matches_patterns(text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern (case-insensitive)"""
        return any(pattern in text for pattern in patterns)


def format_error_for_user(
    classification: ErrorClassification,
    agent_name: str,
    instruction: str,
    attempt_number: int = 1,
    max_attempts: int = 3
) -> str:
    """
    Format an error classification into a user-friendly message.

    Args:
        classification: The error classification
        agent_name: Name of the agent that failed
        instruction: The original instruction
        attempt_number: Current attempt number
        max_attempts: Maximum retry attempts

    Returns:
        Formatted error message for user
    """

    # Build error header based on category
    if classification.category == ErrorCategory.CAPABILITY:
        header = "❌ **Cannot perform this operation**"
    elif classification.category == ErrorCategory.PERMISSION:
        header = "🔐 **Access Denied**"
    elif classification.category == ErrorCategory.RATE_LIMIT:
        header = "⏳ **Rate Limited - Retrying**"
    elif classification.category == ErrorCategory.TRANSIENT:
        header = "⏳ **Temporary Issue - Retrying**"
    elif classification.category == ErrorCategory.VALIDATION:
        header = "❌ **Invalid Input**"
    else:
        header = "⚠️ **Error**"

    # Build message
    message = f"{header}\n\n"
    message += f"**What happened**: {classification.explanation}\n\n"

    # Add retry context if applicable
    if classification.is_retryable and attempt_number > 1:
        message += f"**Attempt**: {attempt_number}/{max_attempts}\n\n"

    # Add suggestions
    if classification.suggestions:
        message += "**What you can try**:\n"
        for suggestion in classification.suggestions:
            message += f"{suggestion}\n"
        message += "\n"

    # Add technical details if verbose or UNKNOWN
    if classification.category == ErrorCategory.UNKNOWN:
        message += f"**Technical details**: {classification.technical_details}\n"

    return message.strip()
