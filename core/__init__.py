"""Core System - Backward compatibility imports"""

# Error handling and classification
from .errors import (
    ErrorCategory,
    ErrorClassification,
    ErrorClassifier,
    format_error_for_user,
    DuplicateOperationDetector,
    EnhancedError,
    ErrorMessageEnhancer
)

# Resilience and retry management
from .resilience import (
    RetryAttempt,
    RetryContext,
    RetryManager
)

# User preferences and analytics
from .user import (
    AgentMetrics,
    SessionMetrics,
    AnalyticsCollector,
    AgentPreference,
    CommunicationStyle,
    WorkingHours,
    UserPreferenceManager
)

__all__ = [
    # Error handling
    'ErrorCategory',
    'ErrorClassification',
    'ErrorClassifier',
    'format_error_for_user',
    'DuplicateOperationDetector',
    'EnhancedError',
    'ErrorMessageEnhancer',
    # Resilience
    'RetryAttempt',
    'RetryContext',
    'RetryManager',
    # User & Analytics
    'AgentMetrics',
    'SessionMetrics',
    'AnalyticsCollector',
    'AgentPreference',
    'CommunicationStyle',
    'WorkingHours',
    'UserPreferenceManager',
]
