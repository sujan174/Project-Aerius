"""
Intelligence System v5.0

Modern hybrid intelligence with LLM-based classification.
Legacy keyword-based components removed in favor of semantic understanding.
"""

from .base_types import (
    Intent, IntentType, Entity, EntityType,
    Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)

from .pipeline import (
    TaskDecomposer,
    ConfidenceScorer
)

from .system import (
    ConversationContextManager,
    IntelligentCache,
    CacheKeyBuilder
)

# Modern hybrid intelligence components
from .hybrid_system import HybridIntelligenceSystem
from .llm_classifier import LLMIntentClassifier
from .fast_filter import FastKeywordFilter

__all__ = [
    # Base types
    'Intent', 'IntentType', 'Entity', 'EntityType',
    'Task', 'ExecutionPlan', 'DependencyGraph',
    'Confidence', 'ConfidenceLevel',
    # Task planning (still needed)
    'TaskDecomposer',
    'ConfidenceScorer',
    # Context management (use dependency injection, not global singleton)
    'ConversationContextManager',
    'IntelligentCache',
    'CacheKeyBuilder',
    # Modern hybrid intelligence
    'HybridIntelligenceSystem',
    'LLMIntentClassifier',
    'FastKeywordFilter',
]

__version__ = '5.0.0'
