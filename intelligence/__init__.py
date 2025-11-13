"""
Intelligence Module - Advanced AI Orchestration Intelligence

This module provides sophisticated intelligence components that transform
the orchestrator from a simple delegation system into an intelligent,
adaptive, learning coordination engine.

Components:
- Intent Classification: Understand what users really want
- Entity Extraction: Extract structured information from natural language
- Task Decomposition: Break complex tasks into optimal execution plans
- Confidence Scoring: Make decisions with awareness of certainty
- Context Management: Deep conversation and workspace understanding
- Intelligent Caching: Performance optimization through caching
- Intelligence Coordinator: Central pipeline orchestration

Architecture:
- pipeline.py: Core intelligence processing (intent, entity, task, confidence)
- system.py: System infrastructure (context, cache, coordinator)
- base_types.py: Shared data structures and types

Author: AI System
Version: 4.0 - Consolidated structure
"""

# Base types
from .base_types import (
    Intent, IntentType, Entity, EntityType,
    Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)

# Pipeline components (from pipeline.py)
from .pipeline import (
    IntentClassifier,
    EntityExtractor,
    TaskDecomposer,
    ConfidenceScorer
)

# System components (from system.py)
from .system import (
    ConversationContextManager,
    IntelligentCache,
    CacheKeyBuilder,
    IntelligenceCoordinator,
    get_global_cache,
    configure_global_cache
)

__all__ = [
    # Base types
    'Intent', 'IntentType', 'Entity', 'EntityType',
    'Task', 'ExecutionPlan', 'DependencyGraph',
    'Confidence', 'ConfidenceLevel',

    # Pipeline components
    'IntentClassifier',
    'EntityExtractor',
    'TaskDecomposer',
    'ConfidenceScorer',

    # System components
    'ConversationContextManager',
    'IntelligentCache',
    'CacheKeyBuilder',
    'IntelligenceCoordinator',
    'get_global_cache',
    'configure_global_cache',
]

__version__ = '4.0.0'
