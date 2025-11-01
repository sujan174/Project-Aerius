"""
Base Types for Intelligence System

Defines core data structures used across all intelligence components.

Author: AI System
Version: 2.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from datetime import datetime


# ============================================================================
# INTENT TYPES
# ============================================================================

class IntentType(Enum):
    """Types of user intents"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    ANALYZE = "analyze"
    COORDINATE = "coordinate"
    WORKFLOW = "workflow"
    SEARCH = "search"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Represents a user intent"""
    type: IntentType
    confidence: float  # 0.0 to 1.0
    entities: List['Entity'] = field(default_factory=list)
    implicit_requirements: List[str] = field(default_factory=list)
    raw_indicators: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.type.value}({self.confidence:.2f})"


# ============================================================================
# ENTITY TYPES
# ============================================================================

class EntityType(Enum):
    """Types of entities that can be extracted"""
    PROJECT = "project"
    PERSON = "person"
    TEAM = "team"
    RESOURCE = "resource"
    DATE = "date"
    PRIORITY = "priority"
    STATUS = "status"
    LABEL = "label"
    ISSUE = "issue"
    PR = "pr"
    CHANNEL = "channel"
    REPOSITORY = "repository"
    FILE = "file"
    CODE = "code"
    UNKNOWN = "unknown"


@dataclass
class Entity:
    """Represents an extracted entity"""
    type: EntityType
    value: str
    confidence: float  # 0.0 to 1.0
    context: Optional[str] = None
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.type.value}:{self.value}({self.confidence:.2f})"


# ============================================================================
# TASK TYPES
# ============================================================================

@dataclass
class Task:
    """Represents a decomposed task"""
    id: str
    action: str
    agent: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    conditions: Optional[str] = None
    priority: int = 0
    estimated_duration: float = 0.0  # seconds
    estimated_cost: float = 0.0  # tokens or API cost
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"Task({self.id}: {self.agent or '?'}.{self.action})"


@dataclass
class DependencyGraph:
    """Represents task dependencies"""
    tasks: Dict[str, Task] = field(default_factory=dict)
    edges: List[tuple] = field(default_factory=list)  # (from_id, to_id)

    def add_task(self, task: Task):
        """Add a task to the graph"""
        self.tasks[task.id] = task

    def add_dependency(self, from_task_id: str, to_task_id: str):
        """Add a dependency edge"""
        self.edges.append((from_task_id, to_task_id))

    def get_execution_order(self) -> List[Task]:
        """Get tasks in topologically sorted order"""
        # Simple topological sort
        in_degree = {task_id: 0 for task_id in self.tasks}
        for from_id, to_id in self.edges:
            in_degree[to_id] = in_degree.get(to_id, 0) + 1

        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            task_id = queue.pop(0)
            result.append(self.tasks[task_id])

            # Reduce in-degree for dependent tasks
            for from_id, to_id in self.edges:
                if from_id == task_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)

        return result

    def has_cycle(self) -> bool:
        """Check if graph has cycles"""
        visited = set()
        rec_stack = set()

        def has_cycle_util(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            for from_id, to_id in self.edges:
                if from_id == task_id:
                    if to_id not in visited:
                        if has_cycle_util(to_id):
                            return True
                    elif to_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle_util(task_id):
                    return True

        return False


@dataclass
class ExecutionPlan:
    """Complete execution plan"""
    tasks: List[Task] = field(default_factory=list)
    dependency_graph: Optional[DependencyGraph] = None
    estimated_duration: float = 0.0
    estimated_cost: float = 0.0
    risks: List[str] = field(default_factory=list)
    optimizations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_execution_order(self) -> List[Task]:
        """Get tasks in optimal execution order"""
        if self.dependency_graph:
            return self.dependency_graph.get_execution_order()
        return self.tasks


# ============================================================================
# CONFIDENCE TYPES
# ============================================================================

class ConfidenceLevel(Enum):
    """Confidence levels for decision making"""
    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"            # > 0.8
    MEDIUM = "medium"        # > 0.6
    LOW = "low"              # > 0.4
    VERY_LOW = "very_low"    # <= 0.4


@dataclass
class Confidence:
    """Represents confidence in a decision"""
    score: float  # 0.0 to 1.0
    level: ConfidenceLevel
    factors: Dict[str, float] = field(default_factory=dict)
    uncertainties: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)

    @staticmethod
    def from_score(score: float, factors: Optional[Dict[str, float]] = None) -> 'Confidence':
        """Create Confidence from score"""
        if score > 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif score > 0.8:
            level = ConfidenceLevel.HIGH
        elif score > 0.6:
            level = ConfidenceLevel.MEDIUM
        elif score > 0.4:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW

        return Confidence(
            score=score,
            level=level,
            factors=factors or {}
        )

    def should_proceed(self) -> bool:
        """Should proceed without asking questions?"""
        return self.score > 0.8

    def should_confirm(self) -> bool:
        """Should confirm with user?"""
        return 0.5 < self.score <= 0.8

    def should_clarify(self) -> bool:
        """Should ask clarifying questions?"""
        return self.score <= 0.5

    def __str__(self) -> str:
        return f"{self.level.value}({self.score:.2f})"


# ============================================================================
# CONTEXT TYPES
# ============================================================================

@dataclass
class ConversationTurn:
    """Represents a single conversation turn"""
    role: str  # 'user' or 'assistant'
    message: str
    timestamp: datetime
    intents: List[Intent] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    tasks_executed: List[str] = field(default_factory=list)


@dataclass
class TrackedEntity:
    """Entity being tracked across conversation"""
    entity: Entity
    first_mentioned: datetime
    last_referenced: datetime
    mention_count: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    relationships: List[tuple] = field(default_factory=list)  # (relation_type, other_entity_id)

    def is_recent(self, max_age_seconds: float = 300) -> bool:
        """Is this entity recently referenced?"""
        age = (datetime.now() - self.last_referenced).total_seconds()
        return age <= max_age_seconds


@dataclass
class Pattern:
    """Learned pattern from operations"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    occurrence_count: int = 1
    success_count: int = 0
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorPattern:
    """Pattern of errors"""
    error_type: str
    context_pattern: Dict[str, Any]
    solutions: List[str]
    occurrence_count: int = 1
    success_rate: float = 0.0


# ============================================================================
# AGENT SELECTION TYPES
# ============================================================================

@dataclass
class AgentScore:
    """Score for an agent's suitability for a task"""
    agent_name: str
    total_score: float  # 0.0 to 1.0
    capability_match: float = 0.0
    health_score: float = 0.0
    context_relevance: float = 0.0
    cost_efficiency: float = 0.0
    historical_success: float = 0.0
    reasoning: str = ""

    def __str__(self) -> str:
        return f"{self.agent_name}({self.total_score:.2f})"


# ============================================================================
# OPTIMIZATION TYPES
# ============================================================================

@dataclass
class CostEstimate:
    """Cost estimate for execution"""
    estimated_tokens: int
    estimated_api_calls: int
    estimated_duration_seconds: float
    estimated_cost_usd: float
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Optimization:
    """Suggested optimization"""
    optimization_type: str
    description: str
    estimated_savings: Dict[str, float]  # {'tokens': 100, 'time': 2.5, 'cost': 0.01}
    implementation: str
    confidence: float = 0.8
