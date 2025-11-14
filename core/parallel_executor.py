"""
Parallel Agent Execution System

Analyzes dependencies between agent tasks and executes them in parallel
where possible, significantly improving performance for multi-agent workflows.

Features:
- Dependency analysis (detect when one agent needs another's output)
- Parallel execution with asyncio.gather()
- Topological ordering for dependent tasks
- Error isolation (one failure doesn't stop independent tasks)
- Performance tracking
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Awaitable
from enum import Enum


class TaskStatus(Enum):
    """Status of a task in the execution graph"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentTask:
    """Represents a single agent execution task"""
    task_id: str
    agent_name: str
    tool_name: str
    instruction: str
    context: Dict[str, Any]
    args: Dict[str, Any]

    # Execution state
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Dependencies
    dependencies: Set[str] = field(default_factory=set)  # task_ids this depends on
    dependents: Set[str] = field(default_factory=set)     # task_ids that depend on this

    def __hash__(self):
        return hash(self.task_id)

    def elapsed_time(self) -> Optional[float]:
        """Return elapsed time in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class DependencyAnalyzer:
    """
    Analyzes dependencies between agent tasks by looking for:
    - References to previous results (e.g., "KAN-123", "PR #456")
    - Context requirements (e.g., "the issue", "that PR")
    - Sequential patterns (e.g., "after creating", "then notify")
    """

    # Patterns that indicate entity references
    ENTITY_PATTERNS = {
        'jira_issue': r'\b([A-Z]{2,10}-\d+)\b',          # KAN-123
        'pr_number': r'\b(?:PR|pull request)\s*#?(\d+)\b',  # PR #456
        'channel': r'#([\w-]+)',                          # #engineering
        'user': r'@([\w.-]+)',                            # @john
        'file': r'[\w-]+\.(py|js|ts|java|go|rs|cpp)',    # main.py
        'url': r'https?://\S+',                           # URLs
    }

    # Keywords indicating dependencies
    DEPENDENCY_KEYWORDS = [
        'after', 'then', 'once', 'when', 'the', 'that', 'this', 'it',
        'using', 'with', 'from', 'based on', 'related to', 'about'
    ]

    @staticmethod
    def analyze_dependencies(tasks: List[AgentTask]) -> List[AgentTask]:
        """
        Analyze all tasks and populate their dependency relationships.
        Returns the same tasks with dependencies filled in.
        """
        # Extract entities created by each task
        created_entities: Dict[str, Set[str]] = {}  # task_id -> set of entity references

        for task in tasks:
            created_entities[task.task_id] = DependencyAnalyzer._extract_entities(
                task.instruction
            )

        # For each task, check if it references entities from previous tasks
        for i, task in enumerate(tasks):
            task_entities = DependencyAnalyzer._extract_entities(task.instruction)
            has_reference_keywords = DependencyAnalyzer._has_reference_keywords(
                task.instruction
            )

            # Check against all previous tasks
            for prev_task in tasks[:i]:
                # Check if this task references entities the previous task will create
                prev_entities = created_entities[prev_task.task_id]

                # Direct entity reference
                if task_entities & prev_entities:
                    task.dependencies.add(prev_task.task_id)
                    prev_task.dependents.add(task.task_id)
                    continue

                # Reference keywords + same agent (likely referring to previous result)
                if has_reference_keywords and task.agent_name == prev_task.agent_name:
                    task.dependencies.add(prev_task.task_id)
                    prev_task.dependents.add(task.task_id)
                    continue

                # Cross-agent patterns (e.g., Jira → Slack notification)
                if DependencyAnalyzer._has_cross_agent_dependency(prev_task, task):
                    task.dependencies.add(prev_task.task_id)
                    prev_task.dependents.add(task.task_id)

        return tasks

    @staticmethod
    def _extract_entities(text: str) -> Set[str]:
        """Extract entity references from text"""
        entities = set()
        text_lower = text.lower()

        for entity_type, pattern in DependencyAnalyzer.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities.update(match)
                else:
                    entities.add(match)

        return entities

    @staticmethod
    def _has_reference_keywords(text: str) -> bool:
        """Check if text contains reference keywords"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in DependencyAnalyzer.DEPENDENCY_KEYWORDS)

    @staticmethod
    def _has_cross_agent_dependency(prev_task: AgentTask, current_task: AgentTask) -> bool:
        """
        Detect common cross-agent dependency patterns:
        - Jira creates issue → Slack notifies about it
        - GitHub creates PR → Jira links to it
        - Any agent creates resource → Another agent references it
        """
        # Common patterns
        cross_agent_patterns = [
            # Previous creates, current notifies
            (r'\b(create|add|make|new)\b', r'\b(notify|post|send|message|tell)\b'),
            # Previous creates, current updates
            (r'\b(create|add|make|new)\b', r'\b(update|link|attach|associate)\b'),
            # Previous searches, current acts on results
            (r'\b(search|find|get|list)\b', r'\b(update|delete|modify)\b'),
        ]

        prev_lower = prev_task.instruction.lower()
        curr_lower = current_task.instruction.lower()

        for prev_pattern, curr_pattern in cross_agent_patterns:
            if re.search(prev_pattern, prev_lower) and re.search(curr_pattern, curr_lower):
                # Different agents and matching pattern = likely dependency
                if prev_task.agent_name != current_task.agent_name:
                    return True

        return False


class ParallelExecutor:
    """
    Executes agent tasks in parallel while respecting dependencies.

    Uses topological sorting to determine execution order:
    - Tasks with no dependencies run immediately in parallel
    - Tasks with dependencies wait for their dependencies to complete
    - Independent tasks at any level run in parallel
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.tasks: Dict[str, AgentTask] = {}
        self.execution_stats = {
            'total_tasks': 0,
            'parallel_executions': 0,
            'sequential_executions': 0,
            'total_time': 0.0,
            'potential_sequential_time': 0.0
        }

    async def execute_tasks(
        self,
        tasks: List[AgentTask],
        executor_func: Callable[[AgentTask], Awaitable[str]]
    ) -> Dict[str, str]:
        """
        Execute all tasks with optimal parallelization.

        Args:
            tasks: List of agent tasks to execute
            executor_func: Async function that executes a single task
                          Should accept AgentTask and return result string

        Returns:
            Dict mapping task_id to result string
        """
        if not tasks:
            return {}

        start_time = time.time()

        # Analyze dependencies
        tasks = DependencyAnalyzer.analyze_dependencies(tasks)
        self.tasks = {task.task_id: task for task in tasks}

        if self.verbose:
            self._print_execution_plan(tasks)

        # Get execution levels (topological sort)
        levels = self._get_execution_levels(tasks)

        results = {}

        # Execute each level in parallel
        for level_num, level_tasks in enumerate(levels):
            if self.verbose:
                print(f"\n🔹 Executing Level {level_num + 1} ({len(level_tasks)} task(s) in parallel)")

            # Execute all tasks in this level in parallel
            level_start = time.time()
            level_results = await self._execute_level(level_tasks, executor_func)
            level_time = time.time() - level_start

            # Update results
            results.update(level_results)

            # Update stats
            if len(level_tasks) > 1:
                self.execution_stats['parallel_executions'] += len(level_tasks)
                if self.verbose:
                    sequential_time = sum(t.elapsed_time() or 0 for t in level_tasks)
                    speedup = sequential_time / level_time if level_time > 0 else 1
                    print(f"   ⚡ Parallel speedup: {speedup:.1f}x ({sequential_time:.1f}s → {level_time:.1f}s)")
            else:
                self.execution_stats['sequential_executions'] += 1

        # Calculate stats
        total_time = time.time() - start_time
        self.execution_stats['total_time'] = total_time
        self.execution_stats['total_tasks'] = len(tasks)
        self.execution_stats['potential_sequential_time'] = sum(
            task.elapsed_time() or 0 for task in tasks
        )

        if self.verbose:
            self._print_execution_stats()

        return results

    async def _execute_level(
        self,
        tasks: List[AgentTask],
        executor_func: Callable[[AgentTask], Awaitable[str]]
    ) -> Dict[str, str]:
        """Execute all tasks in a level in parallel"""

        async def execute_single(task: AgentTask) -> tuple[str, str]:
            """Execute single task and return (task_id, result)"""
            task.status = TaskStatus.RUNNING
            task.start_time = time.time()

            try:
                result = await executor_func(task)
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.end_time = time.time()

                if self.verbose:
                    elapsed = task.elapsed_time()
                    print(f"   ✓ {task.agent_name}: {elapsed:.2f}s")

                return (task.task_id, result)

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.end_time = time.time()

                if self.verbose:
                    print(f"   ✗ {task.agent_name}: {str(e)}")

                # Return error as result so execution continues
                return (task.task_id, f"Error: {str(e)}")

        # Execute all tasks in parallel
        results = await asyncio.gather(
            *[execute_single(task) for task in tasks],
            return_exceptions=False  # Let errors propagate as results
        )

        return dict(results)

    def _get_execution_levels(self, tasks: List[AgentTask]) -> List[List[AgentTask]]:
        """
        Perform topological sort to get execution levels.
        Each level contains tasks that can run in parallel.
        """
        # Create dependency graph
        in_degree = {task.task_id: len(task.dependencies) for task in tasks}
        task_map = {task.task_id: task for task in tasks}

        levels = []
        remaining = set(in_degree.keys())

        while remaining:
            # Find all tasks with no remaining dependencies
            current_level = [
                task_map[task_id]
                for task_id in remaining
                if in_degree[task_id] == 0
            ]

            if not current_level:
                # Circular dependency detected - break it
                # Take task with minimum dependencies
                min_task_id = min(remaining, key=lambda tid: in_degree[tid])
                current_level = [task_map[min_task_id]]
                if self.verbose:
                    print(f"⚠️  Warning: Possible circular dependency detected, breaking with {min_task_id}")

            levels.append(current_level)

            # Remove current level from graph
            for task in current_level:
                remaining.remove(task.task_id)

                # Decrease in-degree for dependents
                for dependent_id in task.dependents:
                    if dependent_id in in_degree:
                        in_degree[dependent_id] -= 1

        return levels

    def _print_execution_plan(self, tasks: List[AgentTask]):
        """Print the execution plan showing dependencies"""
        print("\n📋 Execution Plan:")
        print("=" * 60)

        for i, task in enumerate(tasks):
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else " (no dependencies)"
            print(f"{i+1}. {task.agent_name}: {task.instruction[:60]}...{deps}")

        print("=" * 60)

    def _print_execution_stats(self):
        """Print execution statistics"""
        stats = self.execution_stats
        potential_time = stats['potential_sequential_time']
        actual_time = stats['total_time']

        if potential_time > 0:
            speedup = potential_time / actual_time
        else:
            speedup = 1.0

        print("\n" + "=" * 60)
        print("📊 Parallel Execution Statistics:")
        print(f"   Total tasks: {stats['total_tasks']}")
        print(f"   Parallel executions: {stats['parallel_executions']}")
        print(f"   Sequential executions: {stats['sequential_executions']}")
        print(f"   Potential sequential time: {potential_time:.2f}s")
        print(f"   Actual parallel time: {actual_time:.2f}s")
        print(f"   ⚡ Overall speedup: {speedup:.1f}x")
        print("=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics"""
        return self.execution_stats.copy()
