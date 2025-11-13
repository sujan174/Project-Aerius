"""
Intelligence System Infrastructure

This module consolidates the system-level intelligence components:
- Conversation Context Management: Multi-turn conversation tracking
- Intelligent Caching: LRU cache with TTL for expensive operations
- Intelligence Coordinator: Central orchestration pipeline

These components provide the infrastructure that supports the
core intelligence pipeline (intent, entity, task, confidence).

Merged from:
- context_manager.py
- cache_layer.py
- coordinator.py

Author: AI System
Version: 4.0 - Consolidated system infrastructure
"""

from typing import Any, Optional, Dict, List, Callable, Tuple, Set
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, field
import threading
import time

from .base_types import (
    ConversationTurn, TrackedEntity, Entity, EntityType,
    Intent, Pattern, CacheEntry, hash_content,
    PipelineContext, ProcessingStage, ProcessingResult,
    PerformanceMetrics, QualityMetrics, Confidence, ExecutionPlan
)


# ============================================================================
# CONVERSATION CONTEXT MANAGEMENT
# ============================================================================

class ConversationContextManager:
    """
    Track and maintain conversation context

    Capabilities:
    - Remember conversation history
    - Track entities mentioned across turns
    - Resolve coreferences ("it", "that", "the issue")
    - Maintain topic focus
    - Understand temporal context
    """

    def __init__(self, session_id: str, verbose: bool = False):
        self.session_id = session_id
        self.verbose = verbose

        # Conversation history
        self.turns: List[ConversationTurn] = []

        # Entity tracking
        self.tracked_entities: Dict[str, TrackedEntity] = {}  # entity_id -> TrackedEntity

        # Current focus
        self.current_topic: Optional[str] = None
        self.focused_entities: List[str] = []  # Recently mentioned entity IDs

        # Temporal context
        self.current_project: Optional[str] = None
        self.current_repository: Optional[str] = None
        self.current_branch: Optional[str] = None

        # Learned patterns
        self.patterns: List[Pattern] = []

    def add_turn(
        self,
        role: str,
        message: str,
        intents: Optional[List[Intent]] = None,
        entities: Optional[List[Entity]] = None,
        tasks_executed: Optional[List[str]] = None
    ):
        """Add a conversation turn"""
        turn = ConversationTurn(
            role=role,
            message=message,
            timestamp=datetime.now(),
            intents=intents or [],
            entities=entities or [],
            tasks_executed=tasks_executed or []
        )

        self.turns.append(turn)

        # Track entities from this turn
        if entities:
            self._track_entities(entities)

        # Update focus
        if role == 'user':
            self._update_focus(message, entities or [])

        if self.verbose:
            print(f"[CONTEXT] Added {role} turn: {message[:50]}...")
            if entities:
                print(f"  Entities: {[str(e) for e in entities[:3]]}")

    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns"""
        return self.turns[-count:] if self.turns else []

    def resolve_reference(self, phrase: str) -> Optional[Tuple[str, Entity]]:
        """Resolve ambiguous references like 'it', 'that', 'the issue'"""
        phrase_lower = phrase.lower().strip()

        # Exact ambiguous references
        exact_refs = {
            'it', 'that', 'this', 'them', 'those',
            'the issue', 'the ticket', 'the pr',
            'the pull request', 'the page'
        }

        if phrase_lower in exact_refs:
            # Return most recently mentioned entity
            if self.focused_entities:
                entity_id = self.focused_entities[-1]
                if entity_id in self.tracked_entities:
                    tracked = self.tracked_entities[entity_id]
                    return (entity_id, tracked.entity)

        # Type-specific references
        if phrase_lower in ['the issue', 'the ticket']:
            return self._get_most_recent_by_type(EntityType.ISSUE)

        elif phrase_lower in ['the pr', 'the pull request']:
            return self._get_most_recent_by_type(EntityType.PR)

        elif phrase_lower in ['the channel']:
            return self._get_most_recent_by_type(EntityType.CHANNEL)

        return None

    def get_relevant_context(self, current_message: str) -> Dict:
        """Get relevant context for current message"""
        context = {
            'recent_turns': self.get_recent_turns(3),
            'current_project': self.current_project,
            'current_repository': self.current_repository,
            'focused_entities': self._get_focused_entities(),
            'recent_tasks': self._get_recent_tasks(),
            'temporal': {
                'project': self.current_project,
                'repository': self.current_repository,
                'branch': self.current_branch
            }
        }

        return context

    def _track_entities(self, entities: List[Entity]):
        """Track entities across conversation"""
        now = datetime.now()

        for entity in entities:
            # Create entity ID
            entity_id = f"{entity.type.value}:{entity.value}"

            if entity_id in self.tracked_entities:
                # Update existing entity
                tracked = self.tracked_entities[entity_id]
                tracked.last_referenced = now
                tracked.mention_count += 1

            else:
                # Create new tracked entity
                tracked = TrackedEntity(
                    entity=entity,
                    first_mentioned=now,
                    last_referenced=now,
                    mention_count=1
                )
                self.tracked_entities[entity_id] = tracked

            # Add to focus
            if entity_id not in self.focused_entities:
                self.focused_entities.append(entity_id)

        # Keep focus list bounded
        if len(self.focused_entities) > 10:
            self.focused_entities = self.focused_entities[-10:]

        # Update temporal context
        self._update_temporal_context(entities)

    def _update_focus(self, message: str, entities: List[Entity]):
        """Update current focus based on message"""
        # Detect topic changes
        topic_keywords = {
            'authentication': ['auth', 'login', 'password', 'security'],
            'bugs': ['bug', 'issue', 'problem', 'error'],
            'features': ['feature', 'enhancement', 'new'],
            'deployment': ['deploy', 'release', 'production'],
        }

        message_lower = message.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                if self.current_topic != topic:
                    if self.verbose:
                        print(f"[CONTEXT] Topic changed: {self.current_topic} → {topic}")
                    self.current_topic = topic
                break

    def _update_temporal_context(self, entities: List[Entity]):
        """Update temporal context (current project, repo, etc.)"""
        for entity in entities:
            if entity.type == EntityType.PROJECT and entity.confidence > 0.8:
                self.current_project = entity.value

            elif entity.type == EntityType.REPOSITORY and entity.confidence > 0.8:
                self.current_repository = entity.value

    def _get_focused_entities(self) -> List[Dict]:
        """Get currently focused entities with details"""
        focused = []

        for entity_id in reversed(self.focused_entities[-5:]):  # Last 5 focused
            if entity_id in self.tracked_entities:
                tracked = self.tracked_entities[entity_id]

                # Only include recent entities (within last 5 minutes)
                if tracked.is_recent(max_age_seconds=300):
                    focused.append({
                        'type': tracked.entity.type.value,
                        'value': tracked.entity.value,
                        'mentions': tracked.mention_count,
                        'last_seen': tracked.last_referenced
                    })

        return focused

    def _get_recent_tasks(self) -> List[str]:
        """Get recently executed tasks"""
        tasks = []
        for turn in reversed(self.turns[-5:]):
            if turn.tasks_executed:
                tasks.extend(turn.tasks_executed)
        return tasks

    def _get_most_recent_by_type(self, entity_type: EntityType) -> Optional[Tuple[str, Entity]]:
        """Get most recently mentioned entity of specific type"""
        candidates = []

        for entity_id, tracked in self.tracked_entities.items():
            if tracked.entity.type == entity_type and tracked.is_recent():
                candidates.append((tracked.last_referenced, entity_id, tracked.entity))

        if candidates:
            # Sort by most recent
            candidates.sort(reverse=True)
            _, entity_id, entity = candidates[0]
            return (entity_id, entity)

        return None

    def add_entity_relationship(
        self,
        from_entity_id: str,
        relation_type: str,
        to_entity_id: str
    ):
        """Add relationship between entities"""
        if from_entity_id in self.tracked_entities:
            tracked = self.tracked_entities[from_entity_id]
            tracked.relationships.append((relation_type, to_entity_id))

            if self.verbose:
                print(f"[CONTEXT] Relationship: {from_entity_id} --{relation_type}-> {to_entity_id}")

    def get_related_entities(self, entity_id: str) -> List[Tuple[str, str, Entity]]:
        """Get entities related to given entity"""
        if entity_id not in self.tracked_entities:
            return []

        tracked = self.tracked_entities[entity_id]
        related = []

        for relation_type, related_id in tracked.relationships:
            if related_id in self.tracked_entities:
                related_entity = self.tracked_entities[related_id].entity
                related.append((relation_type, related_id, related_entity))

        return related

    def learn_pattern(
        self,
        pattern_type: str,
        pattern_data: Dict,
        success: bool = True
    ):
        """Learn a pattern from user behavior"""
        # Check if pattern exists
        existing = None
        for pattern in self.patterns:
            if pattern.pattern_type == pattern_type:
                # Check if data is similar
                if self._patterns_match(pattern.pattern_data, pattern_data):
                    existing = pattern
                    break

        if existing:
            # Update existing pattern
            existing.occurrence_count += 1
            if success:
                existing.success_count += 1
            existing.last_seen = datetime.now()

        else:
            # Create new pattern
            pattern = Pattern(
                pattern_type=pattern_type,
                pattern_data=pattern_data,
                confidence=0.5,
                occurrence_count=1,
                success_count=1 if success else 0,
                last_seen=datetime.now()
            )
            self.patterns.append(pattern)

        if self.verbose:
            print(f"[CONTEXT] Learned pattern: {pattern_type} (occurrences: {existing.occurrence_count if existing else 1})")

    def get_learned_patterns(self, pattern_type: Optional[str] = None) -> List[Pattern]:
        """Get learned patterns, optionally filtered by type"""
        if pattern_type:
            return [p for p in self.patterns if p.pattern_type == pattern_type]
        return self.patterns

    def _patterns_match(self, pattern1: Dict, pattern2: Dict, threshold: float = 0.7) -> bool:
        """Check if two pattern data dictionaries match sufficiently"""
        keys1 = set(pattern1.keys())
        keys2 = set(pattern2.keys())

        if not keys1 or not keys2:
            return False

        overlap = keys1 & keys2
        match_ratio = len(overlap) / max(len(keys1), len(keys2))

        if match_ratio < threshold:
            return False

        # Check if values for overlapping keys are similar
        matching_values = sum(
            1 for key in overlap
            if str(pattern1.get(key)) == str(pattern2.get(key))
        )

        value_match_ratio = matching_values / len(overlap) if overlap else 0
        return value_match_ratio >= threshold

    def get_context_summary(self) -> str:
        """Get human-readable summary of current context"""
        lines = []
        lines.append(f"Session: {self.session_id}")
        lines.append(f"Turns: {len(self.turns)}")

        if self.current_project:
            lines.append(f"Current Project: {self.current_project}")

        if self.current_repository:
            lines.append(f"Current Repository: {self.current_repository}")

        if self.current_topic:
            lines.append(f"Current Topic: {self.current_topic}")

        focused = self._get_focused_entities()
        if focused:
            lines.append(f"Focused Entities: {len(focused)}")
            for entity in focused[:3]:
                lines.append(f"  - {entity['type']}: {entity['value']}")

        if self.patterns:
            lines.append(f"Learned Patterns: {len(self.patterns)}")

        return "\n".join(lines)


# ============================================================================
# INTELLIGENT CACHING
# ============================================================================

class IntelligentCache:
    """
    LRU Cache with TTL and statistics

    Thread-safe caching layer for expensive intelligence operations.
    Uses LRU eviction and optional TTL for entries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: Optional[float] = 300,  # 5 minutes default
        verbose: bool = False
    ):
        """Initialize cache"""
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self.verbose = verbose

        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired():
                del self._cache[key]
                self.expirations += 1
                self.misses += 1
                if self.verbose:
                    print(f"[CACHE] Key expired: {key[:30]}...")
                return None

            # Touch entry and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)

            self.hits += 1
            if self.verbose:
                print(f"[CACHE] Hit: {key[:30]}... (accessed {entry.access_count} times)")

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None
    ):
        """Set value in cache"""
        with self._lock:
            now = datetime.now()

            # Use provided TTL or default
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=0,
                ttl_seconds=ttl
            )

            # If key exists, remove it (will be re-added at end)
            if key in self._cache:
                del self._cache[key]

            # Add to end (most recently used)
            self._cache[key] = entry

            # Evict oldest if over capacity
            if len(self._cache) > self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self.evictions += 1
                if self.verbose:
                    print(f"[CACHE] Evicted: {oldest_key[:30]}...")

            if self.verbose:
                print(f"[CACHE] Set: {key[:30]}... (TTL: {ttl}s)")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: Optional[float] = None
    ) -> Any:
        """Get from cache or compute and cache"""
        # Try to get from cache
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        if self.verbose:
            print(f"[CACHE] Computing: {key[:30]}...")

        value = compute_fn()

        # Cache computed value
        self.set(key, value, ttl_seconds)

        return value

    def invalidate(self, key: str) -> bool:
        """Invalidate (remove) cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self.verbose:
                    print(f"[CACHE] Invalidated: {key[:30]}...")
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if pattern in key
            ]

            for key in keys_to_remove:
                del self._cache[key]

            if self.verbose and keys_to_remove:
                print(f"[CACHE] Invalidated {len(keys_to_remove)} entries matching: {pattern}")

            return len(keys_to_remove)

    def clear(self):
        """Clear entire cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            if self.verbose:
                print(f"[CACHE] Cleared {count} entries")

    def cleanup_expired(self) -> int:
        """Remove all expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]

            for key in expired_keys:
                del self._cache[key]

            self.expirations += len(expired_keys)

            if self.verbose and expired_keys:
                print(f"[CACHE] Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'expirations': self.expirations,
                'total_requests': total_requests,
            }

    def reset_stats(self):
        """Reset statistics counters"""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.expirations = 0
            if self.verbose:
                print("[CACHE] Statistics reset")

    def __len__(self) -> int:
        """Get number of entries in cache"""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache"""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            return not entry.is_expired()


class CacheKeyBuilder:
    """Helper to build cache keys consistently"""

    @staticmethod
    def for_intent_classification(message: str) -> str:
        """Build cache key for intent classification"""
        return f"intent:{hash_content(message)}"

    @staticmethod
    def for_entity_extraction(message: str) -> str:
        """Build cache key for entity extraction"""
        return f"entity:{hash_content(message)}"

    @staticmethod
    def for_task_decomposition(message: str, intent_types: str) -> str:
        """Build cache key for task decomposition"""
        return f"task:{hash_content(message)}:{hash_content(intent_types)}"

    @staticmethod
    def for_confidence_score(message: str, intents: str, entities: str) -> str:
        """Build cache key for confidence scoring"""
        components = f"{message}|{intents}|{entities}"
        return f"confidence:{hash_content(components)}"

    @staticmethod
    def for_llm_call(prompt: str, model: str) -> str:
        """Build cache key for LLM calls"""
        return f"llm:{model}:{hash_content(prompt)}"

    @staticmethod
    def for_semantic_similarity(text: str) -> str:
        """Build cache key for semantic embeddings"""
        return f"embedding:{hash_content(text)}"


# Global cache instance (can be configured)
_global_cache: Optional[IntelligentCache] = None


def get_global_cache() -> IntelligentCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache(
            max_size=1000,
            default_ttl_seconds=300,  # 5 minutes
            verbose=False
        )
    return _global_cache


def configure_global_cache(
    max_size: int = 1000,
    default_ttl_seconds: Optional[float] = 300,
    verbose: bool = False
):
    """Configure global cache instance"""
    global _global_cache
    _global_cache = IntelligentCache(
        max_size=max_size,
        default_ttl_seconds=default_ttl_seconds,
        verbose=verbose
    )


# ============================================================================
# INTELLIGENCE COORDINATOR
# ============================================================================

class IntelligenceCoordinator:
    """
    Coordinates all intelligence components in a pipeline

    Pipeline Stages:
    1. Preprocessing - Normalize and validate input
    2. Intent Classification - Understand user intent
    3. Entity Extraction - Extract structured information
    4. Context Integration - Integrate conversation context
    5. Task Decomposition - Break down into executable tasks
    6. Confidence Scoring - Score confidence in understanding
    7. Decision Making - Decide on action (proceed/review/clarify)

    Features:
    - Caching of expensive operations
    - Metrics collection at each stage
    - Error handling and graceful degradation
    - Performance optimization
    """

    def __init__(
        self,
        session_id: str,
        agent_capabilities: Optional[Dict[str, List[str]]] = None,
        cache: Optional[IntelligentCache] = None,
        verbose: bool = False
    ):
        """Initialize intelligence coordinator"""
        self.session_id = session_id
        self.agent_capabilities = agent_capabilities or {}
        self.cache = cache or get_global_cache()
        self.verbose = verbose

        # Import pipeline components here to avoid circular imports
        from .pipeline import (
            IntentClassifier, EntityExtractor,
            TaskDecomposer, ConfidenceScorer
        )

        # Initialize components
        self.intent_classifier = IntentClassifier(verbose=verbose)
        self.entity_extractor = EntityExtractor(verbose=verbose)
        self.context_manager = ConversationContextManager(session_id, verbose=verbose)
        self.task_decomposer = TaskDecomposer(agent_capabilities, verbose=verbose)
        self.confidence_scorer = ConfidenceScorer(verbose=verbose)

        # Metrics
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()

        # Processing history
        self.processing_history: List[PipelineContext] = []

        if self.verbose:
            print(f"[COORDINATOR] Initialized for session: {session_id}")

    def process(self, message: str, user_id: Optional[str] = None) -> PipelineContext:
        """Process user message through intelligence pipeline"""
        start_time = time.time()

        # Create pipeline context
        context = PipelineContext(
            message=message,
            session_id=self.session_id,
            user_id=user_id
        )

        try:
            # Stage 1: Preprocessing
            self._stage_preprocessing(context)

            # Stage 2: Intent Classification
            self._stage_intent_classification(context)

            # Stage 3: Entity Extraction
            self._stage_entity_extraction(context)

            # Stage 4: Context Integration
            self._stage_context_integration(context)

            # Stage 5: Task Decomposition
            self._stage_task_decomposition(context)

            # Stage 6: Confidence Scoring
            self._stage_confidence_scoring(context)

            # Stage 7: Decision Making
            self._stage_decision_making(context)

        except Exception as e:
            # Handle pipeline errors gracefully
            error_result = ProcessingResult(
                stage=ProcessingStage.DECISION_MAKING,
                success=False,
                data={},
                latency_ms=0.0,
                errors=[f"Pipeline error: {str(e)}"]
            )
            context.add_result(error_result)

            if self.verbose:
                print(f"[COORDINATOR] Pipeline error: {e}")

        # Update total latency
        total_latency = (time.time() - start_time) * 1000
        self.performance_metrics.total_latency_ms += total_latency

        # Add to history
        self.processing_history.append(context)

        # Keep history bounded
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]

        if self.verbose:
            print(f"[COORDINATOR] Processing complete in {total_latency:.1f}ms")
            self._print_summary(context)

        return context

    def _stage_preprocessing(self, context: PipelineContext):
        """Stage 1: Preprocessing"""
        start_time = time.time()

        # Normalize message
        normalized_message = context.message.strip()

        # Validate message
        errors = []
        warnings = []

        if not normalized_message:
            errors.append("Empty message")

        if len(normalized_message) > 2000:
            warnings.append("Message very long (>2000 chars)")

        result = ProcessingResult(
            stage=ProcessingStage.PREPROCESSING,
            success=len(errors) == 0,
            data={'normalized_message': normalized_message},
            latency_ms=(time.time() - start_time) * 1000,
            errors=errors,
            warnings=warnings
        )

        context.add_result(result)
        context.message = normalized_message

    def _stage_intent_classification(self, context: PipelineContext):
        """Stage 2: Intent Classification"""
        start_time = time.time()

        try:
            # Classify intents
            intents = self.intent_classifier.classify(context.message)

            result = ProcessingResult(
                stage=ProcessingStage.INTENT_CLASSIFICATION,
                success=True,
                data={
                    'intents': intents,
                    'primary_intent': intents[0] if intents else None
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.intents = intents
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.intent_classification_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.INTENT_CLASSIFICATION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_entity_extraction(self, context: PipelineContext):
        """Stage 3: Entity Extraction"""
        start_time = time.time()

        try:
            # Get conversation context for entity extraction
            conv_context = self.context_manager.get_relevant_context(context.message)

            # Extract entities
            entities = self.entity_extractor.extract(context.message, conv_context)

            result = ProcessingResult(
                stage=ProcessingStage.ENTITY_EXTRACTION,
                success=True,
                data={
                    'entities': entities,
                    'entity_count': len(entities)
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.entities = entities
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.entity_extraction_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.ENTITY_EXTRACTION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_context_integration(self, context: PipelineContext):
        """Stage 4: Context Integration"""
        start_time = time.time()

        try:
            # Add turn to context manager
            self.context_manager.add_turn(
                role='user',
                message=context.message,
                intents=context.intents,
                entities=context.entities
            )

            # Get relevant context
            conversation_context = self.context_manager.get_relevant_context(context.message)

            result = ProcessingResult(
                stage=ProcessingStage.CONTEXT_INTEGRATION,
                success=True,
                data={
                    'conversation_context': conversation_context,
                    'recent_turns': len(conversation_context.get('recent_turns', [])),
                    'focused_entities': len(conversation_context.get('focused_entities', []))
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.conversation_context = conversation_context
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.context_integration_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.CONTEXT_INTEGRATION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_task_decomposition(self, context: PipelineContext):
        """Stage 5: Task Decomposition"""
        start_time = time.time()

        try:
            # Decompose into tasks
            execution_plan = self.task_decomposer.decompose(
                message=context.message,
                intents=context.intents,
                entities=context.entities,
                context=context.conversation_context
            )

            result = ProcessingResult(
                stage=ProcessingStage.TASK_DECOMPOSITION,
                success=True,
                data={
                    'execution_plan': execution_plan,
                    'task_count': len(execution_plan.tasks),
                    'estimated_duration': execution_plan.estimated_duration,
                    'estimated_cost': execution_plan.estimated_cost
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.execution_plan = execution_plan
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.task_decomposition_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.TASK_DECOMPOSITION,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_confidence_scoring(self, context: PipelineContext):
        """Stage 6: Confidence Scoring"""
        start_time = time.time()

        try:
            # Score confidence
            confidence = self.confidence_scorer.score_overall(
                message=context.message,
                intents=context.intents,
                entities=context.entities,
                plan=context.execution_plan
            )

            result = ProcessingResult(
                stage=ProcessingStage.CONFIDENCE_SCORING,
                success=True,
                data={
                    'confidence': confidence,
                    'confidence_score': confidence.score,
                    'confidence_level': confidence.level.value,
                    'uncertainties': confidence.uncertainties,
                    'assumptions': confidence.assumptions
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.confidence = confidence
            context.add_result(result)

            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.performance_metrics.confidence_scoring_ms += latency

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.CONFIDENCE_SCORING,
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _stage_decision_making(self, context: PipelineContext):
        """Stage 7: Decision Making"""
        start_time = time.time()

        try:
            confidence = context.confidence

            if not confidence:
                decision = 'clarify'
                reasoning = "No confidence score available"
            elif self.confidence_scorer.should_proceed_automatically(confidence):
                decision = 'proceed'
                reasoning = f"High confidence ({confidence.score:.2f})"
            elif self.confidence_scorer.should_review_with_user(confidence):
                decision = 'review'
                reasoning = f"Medium confidence ({confidence.score:.2f})"
            else:
                decision = 'clarify'
                reasoning = f"Low confidence ({confidence.score:.2f})"

            # Get clarification questions if needed
            clarifications = []
            if decision == 'clarify' and confidence:
                clarifications = self.confidence_scorer.suggest_clarifications(
                    confidence,
                    context.intents
                )

            result = ProcessingResult(
                stage=ProcessingStage.DECISION_MAKING,
                success=True,
                data={
                    'decision': decision,
                    'reasoning': reasoning,
                    'clarifications': clarifications
                },
                latency_ms=(time.time() - start_time) * 1000
            )

            context.add_result(result)

        except Exception as e:
            result = ProcessingResult(
                stage=ProcessingStage.DECISION_MAKING,
                success=False,
                data={'decision': 'clarify', 'reasoning': 'Error in decision making'},
                latency_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
            context.add_result(result)

    def _print_summary(self, context: PipelineContext):
        """Print processing summary"""
        print("\n" + "="*70)
        print("INTELLIGENCE PROCESSING SUMMARY")
        print("="*70)

        # Intents
        if context.intents:
            print(f"\nIntents ({len(context.intents)}):")
            for intent in context.intents[:3]:
                print(f"  • {intent}")

        # Entities
        if context.entities:
            print(f"\nEntities ({len(context.entities)}):")
            for entity in context.entities[:5]:
                print(f"  • {entity}")

        # Execution Plan
        if context.execution_plan:
            plan = context.execution_plan
            print(f"\nExecution Plan:")
            print(f"  • Tasks: {len(plan.tasks)}")
            print(f"  • Duration: {plan.estimated_duration:.1f}s")
            print(f"  • Cost: {plan.estimated_cost:.0f} tokens")
            if plan.risks:
                print(f"  • Risks: {len(plan.risks)}")

        # Confidence
        if context.confidence:
            conf = context.confidence
            print(f"\nConfidence: {conf.level.value.upper()} ({conf.score:.2f})")
            if conf.uncertainties:
                print(f"  • Uncertainties: {len(conf.uncertainties)}")
            if conf.assumptions:
                print(f"  • Assumptions: {len(conf.assumptions)}")

        # Decision
        decision_result = context.get_stage_result(ProcessingStage.DECISION_MAKING)
        if decision_result and decision_result.success:
            decision = decision_result.data.get('decision')
            reasoning = decision_result.data.get('reasoning')
            print(f"\nDecision: {decision.upper()}")
            print(f"  • {reasoning}")

        # Performance
        total_latency = sum(r.latency_ms for r in context.processing_results)
        print(f"\nPerformance:")
        print(f"  • Total latency: {total_latency:.1f}ms")
        for result in context.processing_results:
            print(f"  • {result.stage.value}: {result.latency_ms:.1f}ms")

        print("="*70 + "\n")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.to_dict()

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics"""
        return self.quality_metrics.to_dict()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

    def reset_metrics(self):
        """Reset all metrics"""
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        self.cache.reset_stats()

        if self.verbose:
            print("[COORDINATOR] Metrics reset")

    def get_context_manager(self) -> ConversationContextManager:
        """Get context manager instance"""
        return self.context_manager

    def get_processing_history(self, count: int = 10) -> List[PipelineContext]:
        """Get recent processing history"""
        return self.processing_history[-count:]
