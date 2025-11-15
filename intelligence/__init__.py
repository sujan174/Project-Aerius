"""
Intelligence System v5.0

Modern hybrid intelligence with LLM-based classification.
Reorganized into logical subdirectories for better maintainability.

Structure:
- classification/: Intent & entity classification (hybrid, LLM, fast filter)
- planning/: Task decomposition and confidence scoring
- context/: Conversation context and cache management
- autonomy/: Confidence-based autonomy and risk classification
- base_types.py: Core data types and enums
"""

# Base types (core data structures)
from .base_types import (
    Intent, IntentType, Entity, EntityType,
    Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)

# Classification components (from classification/)
from .classification import (
    HybridIntelligenceSystem,
    LLMIntentClassifier,
    FastKeywordFilter
)

# Planning components (from planning/)
from .planning import (
    TaskDecomposer,
    ConfidenceScorer
)

# Context management (from context/)
from .context import (
    ConversationContextManager,
    IntelligentCache,
    CacheKeyBuilder,
    get_global_cache,
    configure_global_cache
)

# Autonomy components (from autonomy/)
from .autonomy import (
    RiskLevel,
    OperationRiskClassifier
)

__all__ = [
    # Base types
    'Intent', 'IntentType', 'Entity', 'EntityType',
    'Task', 'ExecutionPlan', 'DependencyGraph',
    'Confidence', 'ConfidenceLevel',
    # Classification
    'HybridIntelligenceSystem',
    'LLMIntentClassifier',
    'FastKeywordFilter',
    # Planning
    'TaskDecomposer',
    'ConfidenceScorer',
    # Context management
    'ConversationContextManager',
    'IntelligentCache',
    'CacheKeyBuilder',
    'get_global_cache',
    'configure_global_cache',
    # Autonomy
    'RiskLevel',
    'OperationRiskClassifier',
]

__version__ = '5.0.0'
