"""
User-Centric Systems

Combines analytics, preferences, and confirmation management:
- User analytics and session tracking
- User preference management
- Message confirmation system

Combines:
- analytics.py: Session analytics and reporting
- user_preferences.py: User preference management
- message_confirmation.py: Confirmation system

Author: AI System
Version: 2.0 (Merged)
"""

from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import statistics
import asyncio
import time
import json
import os


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent"""
    agent_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    error_counts: Dict[str, int] = field(default_factory=dict)

    # Latency percentiles (tracked separately)
    latencies: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 - 1.0)"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (0.0 - 1.0)"""
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds"""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def p50_latency_ms(self) -> float:
        """Get 50th percentile (median) latency"""
        if not self.latencies:
            return 0.0
        return statistics.median(self.latencies)

    @property
    def p95_latency_ms(self) -> float:
        """Get 95th percentile latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        """Get 99th percentile latency"""
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


@dataclass
class SessionMetrics:
    """Metrics for a user session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    user_messages: int = 0
    agent_calls: int = 0
    confirmations_shown: int = 0
    confirmations_accepted: int = 0
    errors_encountered: int = 0
    successful_operations: int = 0

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds"""
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def confirmation_acceptance_rate(self) -> float:
        """Calculate confirmation acceptance rate"""
        if self.confirmations_shown == 0:
            return 0.0
        return self.confirmations_accepted / self.confirmations_shown


class AnalyticsCollector:
    """
    Collects and analyzes usage metrics.

    Features:
    - Real-time metric collection
    - Agent performance tracking
    - Session analytics
    - Trend analysis
    - Health monitoring
    """

    def __init__(
        self,
        session_id: str,
        max_latency_samples: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize analytics collector.

        Args:
            session_id: Current session ID
            max_latency_samples: Max latency samples to keep per agent
            verbose: Enable detailed logging
        """
        self.session_id = session_id
        self.max_latency_samples = max_latency_samples
        self.verbose = verbose

        # Agent metrics
        self.agent_metrics: Dict[str, AgentMetrics] = {}

        # Session tracking
        self.current_session = SessionMetrics(
            session_id=session_id,
            start_time=time.time()
        )
        self.session_history: List[SessionMetrics] = []

        # Usage patterns
        self.hourly_usage: Dict[int, int] = defaultdict(int)  # hour -> count
        self.daily_usage: Dict[str, int] = defaultdict(int)   # date -> count

        # Tool/operation tracking
        self.operation_counts: Counter = Counter()
        self.operation_success_counts: Counter = Counter()

        # Error tracking
        self.error_patterns: Counter = Counter()
        self.agent_error_patterns: Dict[str, Counter] = defaultdict(Counter)

    # =========================================================================
    # AGENT METRICS
    # =========================================================================

    def record_agent_call(
        self,
        agent_name: str,
        success: bool,
        latency_ms: float,
        error_message: Optional[str] = None
    ):
        """
        Record an agent call with performance metrics.

        Args:
            agent_name: Name of the agent
            success: Whether call succeeded
            latency_ms: Call duration in milliseconds
            error_message: Error message if failed
        """
        # Initialize metrics for agent if needed
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(agent_name=agent_name)

        metrics = self.agent_metrics[agent_name]

        # Update counts
        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1
            self.current_session.successful_operations += 1
        else:
            metrics.failed_calls += 1
            self.current_session.errors_encountered += 1

        # Update latency
        metrics.total_latency_ms += latency_ms
        metrics.latencies.append(latency_ms)

        # Trim latencies to max samples
        if len(metrics.latencies) > self.max_latency_samples:
            metrics.latencies = metrics.latencies[-self.max_latency_samples:]

        # Track errors
        if error_message:
            # Classify error type
            error_type = self._classify_error(error_message)
            metrics.error_counts[error_type] = metrics.error_counts.get(error_type, 0) + 1

            # Track error patterns
            self.error_patterns[error_type] += 1
            self.agent_error_patterns[agent_name][error_type] += 1

        # Update session metrics
        self.current_session.agent_calls += 1

        if self.verbose:
            status = "✓" if success else "✗"
            print(f"[ANALYTICS] {status} {agent_name}: {latency_ms:.0f}ms")

    def _classify_error(self, error_message: str) -> str:
        """Classify error into categories"""
        error_lower = error_message.lower()

        if 'timeout' in error_lower or 'timed out' in error_lower:
            return 'timeout'
        elif 'permission' in error_lower or 'forbidden' in error_lower or '403' in error_lower:
            return 'permission'
        elif 'not found' in error_lower or '404' in error_lower:
            return 'not_found'
        elif 'rate limit' in error_lower or '429' in error_lower:
            return 'rate_limit'
        elif 'network' in error_lower or 'connection' in error_lower:
            return 'network'
        elif 'validation' in error_lower or 'invalid' in error_lower:
            return 'validation'
        else:
            return 'other'

    def get_agent_metrics(self, agent_name: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent"""
        return self.agent_metrics.get(agent_name)

    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents"""
        return self.agent_metrics.copy()

    # =========================================================================
    # SESSION TRACKING
    # =========================================================================

    def record_user_message(self):
        """Record a user message"""
        self.current_session.user_messages += 1

        # Track usage by hour
        current_hour = datetime.now().hour
        self.hourly_usage[current_hour] += 1

        # Track usage by day
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_usage[today] += 1

    def record_confirmation(self, accepted: bool):
        """Record a confirmation prompt"""
        self.current_session.confirmations_shown += 1
        if accepted:
            self.current_session.confirmations_accepted += 1

    def record_operation(self, operation_type: str, success: bool):
        """Record an operation execution"""
        self.operation_counts[operation_type] += 1
        if success:
            self.operation_success_counts[operation_type] += 1

    def end_session(self):
        """End current session and archive metrics"""
        self.current_session.end_time = time.time()
        self.session_history.append(self.current_session)

        # Start new session
        self.current_session = SessionMetrics(
            session_id=f"{self.session_id}_cont",
            start_time=time.time()
        )

        if self.verbose:
            print(f"[ANALYTICS] Session ended: {self.current_session.duration_seconds:.0f}s")

    # =========================================================================
    # ANALYTICS AND REPORTING
    # =========================================================================

    def get_agent_ranking(self) -> List[tuple]:
        """
        Get agents ranked by success rate.

        Returns:
            List of (agent_name, success_rate, total_calls) sorted by success rate
        """
        rankings = [
            (
                agent_name,
                metrics.success_rate,
                metrics.total_calls
            )
            for agent_name, metrics in self.agent_metrics.items()
            if metrics.total_calls > 0
        ]

        # Sort by success rate (descending), then by call count
        rankings.sort(key=lambda x: (-x[1], -x[2]))

        return rankings

    def get_slowest_agents(self, limit: int = 5) -> List[tuple]:
        """
        Get slowest agents by average latency.

        Returns:
            List of (agent_name, avg_latency_ms, call_count)
        """
        slowest = [
            (
                agent_name,
                metrics.avg_latency_ms,
                metrics.total_calls
            )
            for agent_name, metrics in self.agent_metrics.items()
            if metrics.total_calls > 0
        ]

        slowest.sort(key=lambda x: -x[1])

        return slowest[:limit]

    def get_most_used_operations(self, limit: int = 10) -> List[tuple]:
        """Get most frequently used operations"""
        return self.operation_counts.most_common(limit)

    def get_most_common_errors(self, limit: int = 10) -> List[tuple]:
        """Get most common error types"""
        return self.error_patterns.most_common(limit)

    def get_usage_by_hour(self) -> Dict[int, int]:
        """Get usage distribution by hour of day"""
        return dict(self.hourly_usage)

    def get_peak_usage_hours(self, top_n: int = 3) -> List[int]:
        """Get peak usage hours"""
        sorted_hours = sorted(
            self.hourly_usage.items(),
            key=lambda x: -x[1]
        )
        return [hour for hour, _ in sorted_hours[:top_n]]

    def get_health_score(self) -> float:
        """
        Calculate overall system health score (0.0 - 1.0).

        Factors:
        - Overall success rate
        - Agent availability
        - Error rate
        """
        if not self.agent_metrics:
            return 1.0

        # Calculate weighted metrics
        total_calls = sum(m.total_calls for m in self.agent_metrics.values())
        if total_calls == 0:
            return 1.0

        # Success rate score (70% weight)
        total_success = sum(m.successful_calls for m in self.agent_metrics.values())
        success_score = total_success / total_calls

        # Agent availability score (20% weight)
        # Agents with 0 calls are considered unavailable
        available_agents = sum(1 for m in self.agent_metrics.values() if m.total_calls > 0)
        total_agents = len(self.agent_metrics)
        availability_score = available_agents / total_agents if total_agents > 0 else 1.0

        # Error diversity score (10% weight)
        # Lower is better - many different errors suggests systemic issues
        error_types = len(self.error_patterns)
        error_diversity_score = 1.0 - min(error_types / 10, 1.0)

        health_score = (
            success_score * 0.7 +
            availability_score * 0.2 +
            error_diversity_score * 0.1
        )

        return health_score

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_summary_report(self) -> str:
        """Generate human-readable summary report"""
        lines = [
            "📊 **System Analytics Summary**\n",
            f"Session: {self.session_id}",
            f"Duration: {self.current_session.duration_seconds:.0f}s",
            f"Health Score: {self.get_health_score():.1%}\n"
        ]

        # Agent performance
        lines.append("**Agent Performance:**")
        for agent_name, success_rate, calls in self.get_agent_ranking():
            metrics = self.agent_metrics[agent_name]
            lines.append(
                f"  • {agent_name}: {success_rate:.1%} success "
                f"({calls} calls, {metrics.avg_latency_ms:.0f}ms avg)"
            )
        lines.append("")

        # Usage statistics
        lines.append("**Usage Statistics:**")
        lines.append(f"  • User messages: {self.current_session.user_messages}")
        lines.append(f"  • Agent calls: {self.current_session.agent_calls}")
        lines.append(f"  • Successful operations: {self.current_session.successful_operations}")
        lines.append(f"  • Errors: {self.current_session.errors_encountered}")
        lines.append("")

        # Top operations
        if self.operation_counts:
            lines.append("**Most Used Operations:**")
            for op, count in self.get_most_used_operations(5):
                success = self.operation_success_counts[op]
                success_rate = success / count if count > 0 else 0
                lines.append(f"  • {op}: {count} times ({success_rate:.1%} success)")
            lines.append("")

        # Common errors
        if self.error_patterns:
            lines.append("**Common Errors:**")
            for error_type, count in self.get_most_common_errors(5):
                lines.append(f"  • {error_type}: {count} occurrences")
            lines.append("")

        # Peak usage
        peak_hours = self.get_peak_usage_hours(3)
        if peak_hours:
            lines.append(f"**Peak Usage Hours:** {', '.join(f'{h}:00' for h in peak_hours)}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as dictionary"""
        return {
            'session_id': self.session_id,
            'current_session': asdict(self.current_session),
            'agent_metrics': {
                name: {
                    **asdict(metrics),
                    'latencies': metrics.latencies[-100:]  # Keep last 100 only
                }
                for name, metrics in self.agent_metrics.items()
            },
            'hourly_usage': dict(self.hourly_usage),
            'daily_usage': dict(self.daily_usage),
            'operation_counts': dict(self.operation_counts),
            'error_patterns': dict(self.error_patterns),
            'health_score': self.get_health_score(),
            'timestamp': time.time()
        }

    def save_to_file(self, filepath: str):
        """Save analytics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        if self.verbose:
            print(f"[ANALYTICS] Saved to {filepath}")


# ============================================================================
# USER PREFERENCES
# ============================================================================

import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, time as dt_time
from collections import Counter, defaultdict


@dataclass
class ConfirmationPreference:
    """User's learned preference for confirmations"""
    operation_pattern: str  # e.g., "jira_create", "slack_message", "github_delete"
    always_confirm: bool
    auto_execute: bool
    confidence: float = 0.0  # 0.0 - 1.0
    sample_count: int = 0  # How many times we've observed this


@dataclass
class AgentPreference:
    """User's preferred agent for a task type"""
    task_pattern: str  # e.g., "create ticket", "send message"
    preferred_agent: str
    confidence: float = 0.0
    usage_count: int = 0


@dataclass
class CommunicationStyle:
    """User's communication preferences"""
    prefers_verbose: bool = False  # Detailed explanations vs concise
    prefers_technical: bool = True  # Technical details vs simplified
    prefers_emojis: bool = False  # Use emojis in responses

    sample_count: int = 0
    confidence: float = 0.0


@dataclass
class WorkingHours:
    """User's working hours pattern"""
    typical_start_hour: int = 9  # 24-hour format
    typical_end_hour: int = 17
    active_days: Set[int] = field(default_factory=lambda: {0, 1, 2, 3, 4})  # Mon-Fri
    timezone_offset: int = 0  # UTC offset

    sample_count: int = 0
    confidence: float = 0.0


class UserPreferenceManager:
    """
    Learns and manages user preferences over time.

    Features:
    - Implicit learning from user behavior
    - Explicit preference settings
    - Confidence-based recommendations
    - Persistent storage
    """

    def __init__(
        self,
        user_id: str = "default",
        min_confidence_threshold: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize preference manager.

        Args:
            user_id: Unique identifier for this user
            min_confidence_threshold: Minimum confidence to apply learned preferences
            verbose: Enable detailed logging
        """
        self.user_id = user_id
        self.min_confidence_threshold = min_confidence_threshold
        self.verbose = verbose

        # Learned preferences
        self.confirmation_prefs: Dict[str, ConfirmationPreference] = {}
        self.agent_prefs: Dict[str, AgentPreference] = {}
        self.communication_style = CommunicationStyle()
        self.working_hours = WorkingHours()

        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        self.task_patterns: Dict[str, List[str]] = defaultdict(list)  # task_type -> [agent_used]

        # Statistics
        self.total_interactions = 0
        self.confirmations_given = 0
        self.confirmations_rejected = 0
        self.auto_executions = 0

    # =========================================================================
    # CONFIRMATION PREFERENCES
    # =========================================================================

    def record_confirmation_decision(
        self,
        operation_pattern: str,
        user_confirmed: bool,
        had_chance_to_edit: bool = False
    ):
        """
        Record user's decision on a confirmation prompt.

        Args:
            operation_pattern: Type of operation (e.g., "jira_delete", "slack_post")
            user_confirmed: Whether user confirmed or rejected
            had_chance_to_edit: Whether user edited parameters before confirming
        """
        if operation_pattern not in self.confirmation_prefs:
            self.confirmation_prefs[operation_pattern] = ConfirmationPreference(
                operation_pattern=operation_pattern,
                always_confirm=True,  # Start conservative
                auto_execute=False,
                confidence=0.0,
                sample_count=0
            )

        pref = self.confirmation_prefs[operation_pattern]
        pref.sample_count += 1

        if user_confirmed:
            self.confirmations_given += 1
        else:
            self.confirmations_rejected += 1

        # Update preferences based on pattern
        # If user consistently confirms without edits, maybe auto-execute is OK
        if pref.sample_count >= 5:
            confirm_rate = self.confirmations_given / (self.confirmations_given + self.confirmations_rejected)

            if confirm_rate > 0.9 and not had_chance_to_edit:
                # User almost always confirms - consider auto-execute
                pref.auto_execute = True
                pref.always_confirm = False
                pref.confidence = min(0.9, pref.sample_count / 20)
            elif confirm_rate < 0.5:
                # User often rejects - always confirm
                pref.always_confirm = True
                pref.auto_execute = False
                pref.confidence = min(0.9, pref.sample_count / 20)

        if self.verbose:
            print(f"[PREFS] Recorded confirmation: {operation_pattern} -> {'confirmed' if user_confirmed else 'rejected'}")

    def should_auto_execute(self, operation_pattern: str) -> bool:
        """Check if operation can be auto-executed based on learned preferences"""
        if operation_pattern not in self.confirmation_prefs:
            return False

        pref = self.confirmation_prefs[operation_pattern]

        # Only auto-execute if confident
        if pref.confidence >= self.min_confidence_threshold and pref.auto_execute:
            return True

        return False

    def should_always_confirm(self, operation_pattern: str) -> bool:
        """Check if operation should always be confirmed"""
        if operation_pattern not in self.confirmation_prefs:
            return True  # Default to safe side

        pref = self.confirmation_prefs[operation_pattern]
        return pref.always_confirm

    # =========================================================================
    # AGENT PREFERENCES
    # =========================================================================

    def record_agent_usage(
        self,
        task_pattern: str,
        agent_used: str,
        was_successful: bool
    ):
        """
        Record which agent was used for a task.

        Args:
            task_pattern: Type of task (e.g., "create_ticket", "send_message")
            agent_used: Agent that was used
            was_successful: Whether the task succeeded
        """
        # Track patterns
        if was_successful:
            self.task_patterns[task_pattern].append(agent_used)

        # Update agent preference
        if task_pattern not in self.agent_prefs:
            self.agent_prefs[task_pattern] = AgentPreference(
                task_pattern=task_pattern,
                preferred_agent=agent_used,
                confidence=0.0,
                usage_count=0
            )

        pref = self.agent_prefs[task_pattern]
        pref.usage_count += 1

        # Calculate most common agent for this task
        if len(self.task_patterns[task_pattern]) >= 3:
            agent_counts = Counter(self.task_patterns[task_pattern])
            most_common_agent, count = agent_counts.most_common(1)[0]

            pref.preferred_agent = most_common_agent
            pref.confidence = min(0.95, count / len(self.task_patterns[task_pattern]))

        if self.verbose:
            print(f"[PREFS] Recorded agent usage: {task_pattern} -> {agent_used} ({'success' if was_successful else 'failed'})")

    def get_preferred_agent(self, task_pattern: str) -> Optional[str]:
        """Get user's preferred agent for a task pattern"""
        if task_pattern not in self.agent_prefs:
            return None

        pref = self.agent_prefs[task_pattern]

        if pref.confidence >= self.min_confidence_threshold:
            return pref.preferred_agent

        return None

    # =========================================================================
    # COMMUNICATION STYLE
    # =========================================================================

    def record_interaction_style(
        self,
        user_message: str,
        user_requested_verbose: bool = False,
        user_requested_technical: bool = False
    ):
        """
        Learn from user's communication style.

        Args:
            user_message: User's message
            user_requested_verbose: Whether user asked for detailed explanations
            user_requested_technical: Whether user asked for technical details
        """
        self.communication_style.sample_count += 1

        # Update preferences based on explicit requests
        if user_requested_verbose:
            self.communication_style.prefers_verbose = True

        if user_requested_technical:
            self.communication_style.prefers_technical = True

        # Detect emojis in user messages
        if any(char in user_message for char in "😀😃😄😁😆😅🤣😂🙂🙃😉😊"):
            self.communication_style.prefers_emojis = True

        # Update confidence
        if self.communication_style.sample_count >= 5:
            self.communication_style.confidence = min(0.9, self.communication_style.sample_count / 20)

    def get_communication_preferences(self) -> Dict[str, bool]:
        """Get user's communication preferences"""
        if self.communication_style.confidence < self.min_confidence_threshold:
            # Use defaults
            return {
                'verbose': False,
                'technical': True,
                'emojis': False
            }

        return {
            'verbose': self.communication_style.prefers_verbose,
            'technical': self.communication_style.prefers_technical,
            'emojis': self.communication_style.prefers_emojis
        }

    # =========================================================================
    # WORKING HOURS
    # =========================================================================

    def record_interaction_time(self, timestamp: Optional[float] = None):
        """Record time of user interaction to learn working hours"""
        if timestamp is None:
            timestamp = time.time()

        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        weekday = dt.weekday()  # 0 = Monday

        self.working_hours.sample_count += 1

        # Update typical start/end hours (weighted average)
        if self.working_hours.sample_count == 1:
            self.working_hours.typical_start_hour = hour
            self.working_hours.typical_end_hour = hour
        else:
            # Simple running average
            if hour < 12:  # Morning - likely start
                self.working_hours.typical_start_hour = int(
                    (self.working_hours.typical_start_hour + hour) / 2
                )
            elif hour > 15:  # Afternoon/evening - likely end
                self.working_hours.typical_end_hour = int(
                    (self.working_hours.typical_end_hour + hour) / 2
                )

        # Track active days
        self.working_hours.active_days.add(weekday)

        # Update confidence
        if self.working_hours.sample_count >= 10:
            self.working_hours.confidence = min(0.9, self.working_hours.sample_count / 50)

    def is_during_working_hours(self, timestamp: Optional[float] = None) -> bool:
        """Check if timestamp is during user's typical working hours"""
        if self.working_hours.confidence < self.min_confidence_threshold:
            return True  # Assume always OK if not confident

        if timestamp is None:
            timestamp = time.time()

        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour
        weekday = dt.weekday()

        # Check if it's an active day
        if weekday not in self.working_hours.active_days:
            return False

        # Check if it's during working hours
        return self.working_hours.typical_start_hour <= hour <= self.working_hours.typical_end_hour

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert preferences to dictionary for serialization"""
        return {
            'user_id': self.user_id,
            'confirmation_prefs': {
                k: asdict(v) for k, v in self.confirmation_prefs.items()
            },
            'agent_prefs': {
                k: asdict(v) for k, v in self.agent_prefs.items()
            },
            'communication_style': asdict(self.communication_style),
            'working_hours': {
                **asdict(self.working_hours),
                'active_days': list(self.working_hours.active_days)  # Convert set to list
            },
            'statistics': {
                'total_interactions': self.total_interactions,
                'confirmations_given': self.confirmations_given,
                'confirmations_rejected': self.confirmations_rejected,
                'auto_executions': self.auto_executions
            },
            'saved_at': time.time()
        }

    def save_to_file(self, filepath: str):
        """Save preferences to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        if self.verbose:
            print(f"[PREFS] Saved preferences to {filepath}")

    def load_from_file(self, filepath: str):
        """Load preferences from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore confirmation preferences
            self.confirmation_prefs = {
                k: ConfirmationPreference(**v)
                for k, v in data.get('confirmation_prefs', {}).items()
            }

            # Restore agent preferences
            self.agent_prefs = {
                k: AgentPreference(**v)
                for k, v in data.get('agent_prefs', {}).items()
            }

            # Restore communication style
            comm_data = data.get('communication_style', {})
            self.communication_style = CommunicationStyle(**comm_data)

            # Restore working hours
            hours_data = data.get('working_hours', {})
            # Convert active_days list back to set
            if 'active_days' in hours_data:
                hours_data['active_days'] = set(hours_data['active_days'])
            self.working_hours = WorkingHours(**hours_data)

            # Restore statistics
            stats = data.get('statistics', {})
            self.total_interactions = stats.get('total_interactions', 0)
            self.confirmations_given = stats.get('confirmations_given', 0)
            self.confirmations_rejected = stats.get('confirmations_rejected', 0)
            self.auto_executions = stats.get('auto_executions', 0)

            if self.verbose:
                print(f"[PREFS] Loaded preferences from {filepath}")

        except Exception as e:
            if self.verbose:
                print(f"[PREFS] Failed to load preferences: {e}")

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_summary(self) -> str:
        """Get human-readable summary of learned preferences"""
        lines = [
            "📊 **Learned Preferences Summary**\n",
            f"User: {self.user_id}",
            f"Total Interactions: {self.total_interactions}\n"
        ]

        # Confirmation preferences
        if self.confirmation_prefs:
            lines.append("**Confirmation Preferences**:")
            for pattern, pref in self.confirmation_prefs.items():
                status = "Auto-execute" if pref.auto_execute else "Always confirm"
                lines.append(f"  • {pattern}: {status} (confidence: {pref.confidence:.0%})")
            lines.append("")

        # Agent preferences
        if self.agent_prefs:
            lines.append("**Preferred Agents**:")
            for pattern, pref in self.agent_prefs.items():
                if pref.confidence >= self.min_confidence_threshold:
                    lines.append(f"  • {pattern} → {pref.preferred_agent} (confidence: {pref.confidence:.0%})")
            lines.append("")

        # Communication style
        if self.communication_style.confidence >= self.min_confidence_threshold:
            lines.append("**Communication Style**:")
            style = self.get_communication_preferences()
            lines.append(f"  • Verbose: {style['verbose']}")
            lines.append(f"  • Technical: {style['technical']}")
            lines.append(f"  • Emojis: {style['emojis']}")
            lines.append("")

        # Working hours
        if self.working_hours.confidence >= self.min_confidence_threshold:
            lines.append("**Typical Working Hours**:")
            lines.append(f"  • {self.working_hours.typical_start_hour}:00 - {self.working_hours.typical_end_hour}:00")
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            active = ', '.join(days[d] for d in sorted(self.working_hours.active_days))
            lines.append(f"  • Active days: {active}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# MESSAGE CONFIRMATION
# ============================================================================

from dataclasses import dataclass
from enum import Enum

# Import inquirer for interactive terminal UI
try:
    import inquirer
    from inquirer.themes import GreenPassion
    INQUIRER_AVAILABLE = True
except ImportError:
    INQUIRER_AVAILABLE = False

# Import config for toggleable confirmations
try:
    from config import Config
except ImportError:
    # Fallback if config not available
    class Config:
        CONFIRM_SLACK_MESSAGES = True
        CONFIRM_JIRA_OPERATIONS = True


class ConfirmationDecision(str, Enum):
    """User's decision on a message/content"""
    APPROVED = "approved"
    REJECTED = "rejected"
    EDIT_MANUAL = "edit_manual"  # User will edit manually
    EDIT_AI = "edit_ai"  # Ask AI to modify


@dataclass
class MessagePreview:
    """Preview of a message/content before sending"""
    agent_name: str  # slack, notion, etc.
    operation_type: str  # "send_message", "create_page", etc.
    destination: str  # channel name, page title, etc.
    content: str  # The actual message/content
    metadata: Dict[str, Any]  # Additional context

    def format_preview(self) -> str:
        """Format for display to user"""
        lines = [
            f"\n{'='*70}",
            f"📝 **{self.agent_name.upper()} - {self.operation_type}**",
            f"{'='*70}\n",
            f"**Destination:** {self.destination}\n",
            f"**Content Preview:**",
            f"{'-'*70}",
            self.content,
            f"{'-'*70}\n"
        ]

        # Add metadata if present
        if self.metadata:
            lines.append("**Additional Info:**")
            for key, value in self.metadata.items():
                lines.append(f"  • {key}: {value}")
            lines.append("")

        return "\n".join(lines)


class MessageConfirmation:
    """
    Handles mandatory confirmation for Slack/Notion operations.

    Features:
    - Shows preview of message/content
    - Allows manual editing
    - Allows AI-assisted editing
    - Enforces human approval
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def confirm_slack_message(
        self,
        channel: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ConfirmationDecision, Optional[str]]:
        """
        Confirm Slack message before sending.

        Args:
            channel: Target Slack channel
            message: Message content
            metadata: Additional context (thread_ts, etc.)

        Returns:
            (decision, modified_message)
        """
        preview = MessagePreview(
            agent_name="Slack",
            operation_type="Send Message",
            destination=channel,
            content=message,
            metadata=metadata or {}
        )

        return self._confirm_with_edit(preview)

    def confirm_notion_operation(
        self,
        operation_type: str,
        page_title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ConfirmationDecision, Optional[str]]:
        """
        Confirm Notion operation before executing.

        Args:
            operation_type: "Create Page", "Update Page", etc.
            page_title: Title of the page
            content: Page content
            metadata: Additional context

        Returns:
            (decision, modified_content)
        """
        preview = MessagePreview(
            agent_name="Notion",
            operation_type=operation_type,
            destination=page_title,
            content=content,
            metadata=metadata or {}
        )

        return self._confirm_with_edit(preview)

    def _confirm_with_edit(
        self,
        preview: MessagePreview
    ) -> Tuple[ConfirmationDecision, Optional[str]]:
        """
        Main confirmation flow with edit capability.

        Returns:
            (decision, modified_content)
        """
        modified_content = preview.content

        while True:
            # Show preview
            print(preview.format_preview())

            # Show options
            print("**Options:**")
            print("  [a] Approve and send")
            print("  [e] Edit manually")
            print("  [m] Ask AI to modify")
            print("  [r] Reject (don't send)")
            print()

            try:
                choice = input("Your decision [a/e/m/r]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n❌ Cancelled by user")
                return ConfirmationDecision.REJECTED, None

            if choice == 'a':
                # Approved
                print("✅ Approved!")
                return ConfirmationDecision.APPROVED, modified_content

            elif choice == 'e':
                # Manual edit
                print("\n📝 Edit the message below. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) when done:")
                print(f"{'-'*70}")
                print(modified_content)
                print(f"{'-'*70}")

                # Multi-line input
                edited_lines = []
                try:
                    while True:
                        line = input()
                        edited_lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    pass

                if edited_lines:
                    modified_content = '\n'.join(edited_lines)
                    preview.content = modified_content
                    print("\n✅ Message updated! Review below:\n")
                    # Loop back to show updated preview
                else:
                    print("⚠️ No changes made")

            elif choice == 'm':
                # AI-assisted modification
                print("\n🤖 What changes would you like the AI to make?")
                modification_request = input("Your request: ").strip()

                if modification_request:
                    # Return for AI to process
                    return ConfirmationDecision.EDIT_AI, modification_request
                else:
                    print("⚠️ No modification requested")

            elif choice == 'r':
                # Rejected
                print("❌ Rejected - message will not be sent")
                return ConfirmationDecision.REJECTED, None

            else:
                print("⚠️ Invalid choice. Please enter a, e, m, or r")

    def confirm_bulk_messages(
        self,
        messages: List[Tuple[str, str, Dict]]  # (channel, message, metadata)
    ) -> List[Tuple[str, str, bool]]:
        """
        Confirm multiple messages at once.

        Args:
            messages: List of (channel, message, metadata) tuples

        Returns:
            List of (channel, final_message, approved) tuples
        """
        results = []

        print(f"\n🔔 **{len(messages)} messages require confirmation**\n")

        for i, (channel, message, metadata) in enumerate(messages, 1):
            print(f"\n{'='*70}")
            print(f"Message {i}/{len(messages)}")
            print(f"{'='*70}")

            decision, modified = self.confirm_slack_message(
                channel=channel,
                message=message,
                metadata=metadata
            )

            if decision == ConfirmationDecision.APPROVED:
                results.append((channel, modified, True))
            elif decision == ConfirmationDecision.EDIT_AI:
                # AI modification requested - mark for reprocessing
                results.append((channel, modified, False))  # False = needs reprocessing
            else:
                # Rejected
                results.append((channel, message, False))

        return results


class MandatoryConfirmationEnforcer:
    """
    Enforces mandatory confirmation for Slack and Notion.

    This is a safety layer that ensures NO Slack message or Notion
    operation executes without human approval.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.confirmer = MessageConfirmation(verbose=verbose)

        # Track operations that require confirmation
        self.pending_confirmations: List[Dict[str, Any]] = []

        # Store agent instances for accessing prefetched data
        self.agent_instances: Dict[str, Any] = {}

    def requires_confirmation(self, agent_name: str, instruction: str) -> bool:
        """
        Check if this operation requires confirmation.

        Args:
            agent_name: Name of the agent
            instruction: The instruction being executed

        Returns:
            True if confirmation is required
        """
        agent_lower = agent_name.lower()

        # Slack: Only confirm WRITE operations (sending messages, not reading)
        if agent_lower == 'slack' and Config.CONFIRM_SLACK_MESSAGES:
            # Write operations (need confirmation)
            write_keywords = ['send', 'post', 'message to', 'notify', 'announce', 'reply', 'react', 'delete']
            # Read operations (no confirmation needed)
            read_keywords = ['list', 'get', 'search', 'find', 'show', 'view', 'read', 'channels', 'users']

            # If it's a read operation, don't confirm
            if any(read_kw in instruction.lower() for read_kw in read_keywords):
                return False

            # If it's a write operation, confirm
            if any(write_kw in instruction.lower() for write_kw in write_keywords):
                return True

        # Notion: Only confirm WRITE operations (not reads)
        if agent_lower == 'notion':
            # Write operations (need confirmation)
            write_keywords = ['create', 'add', 'update', 'write', 'insert', 'delete', 'edit']
            # Read operations (no confirmation needed)
            read_keywords = ['get', 'search', 'list', 'find', 'show', 'view', 'read', 'pages', 'database']

            # If it's a read operation, don't confirm
            if any(read_kw in instruction.lower() for read_kw in read_keywords):
                return False

            # If it's a write operation, confirm
            if any(write_kw in instruction.lower() for write_kw in write_keywords):
                return True

        # Jira: Only confirm WRITE operations (not reads)
        if agent_lower == 'jira' and Config.CONFIRM_JIRA_OPERATIONS:
            # Only confirm write operations
            write_keywords = ['create', 'update', 'delete', 'transition', 'assign', 'add comment', 'close', 'edit']
            # Exclude read operations
            read_keywords = ['get', 'search', 'list', 'find', 'show', 'view', 'assigned to me', 'my tasks']

            # If it's a read operation, don't confirm
            if any(read_kw in instruction.lower() for read_kw in read_keywords):
                return False

            # If it's a write operation, confirm
            if any(write_kw in instruction.lower() for write_kw in write_keywords):
                return True

        return False

    def extract_message_content(
        self,
        agent_name: str,
        instruction: str
    ) -> Optional[Tuple[str, str, Dict]]:
        """
        Extract message content from instruction.

        Args:
            agent_name: Name of the agent
            instruction: The instruction

        Returns:
            (destination, content, metadata) or None
        """
        agent_lower = agent_name.lower()
        instruction_lower = instruction.lower()

        if agent_lower == 'slack':
            # Try to extract channel and message
            import re

            # Pattern: "send/post ... to #channel"
            channel_match = re.search(r'#([\w\-]+)', instruction)
            channel = channel_match.group(1) if channel_match else "unknown"

            # Extract message content (quoted strings first)
            message_patterns = [
                r'message[:\s]+"([^"]+)"',
                r'message[:\s]+\'([^\']+)\'',
                r'send[:\s]+"([^"]+)"',
                r'send[:\s]+\'([^\']+)\'',
            ]

            message = None
            for pattern in message_patterns:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    message = match.group(1)
                    break

            if not message:
                # Try to extract unquoted message: "send MESSAGE to/on CHANNEL"
                # Match: send/post [MESSAGE] to/on [CHANNEL]
                unquoted_patterns = [
                    r'(?:send|post)\s+(.+?)\s+(?:to|on)\s+(?:#?[\w\-\s]+channel|#[\w\-]+|slack)',
                    r'(?:send|post)\s+(.+?)\s+(?:to|on)',
                ]

                for pattern in unquoted_patterns:
                    match = re.search(pattern, instruction, re.IGNORECASE)
                    if match:
                        message = match.group(1).strip()
                        # Clean up common artifacts
                        message = re.sub(r'\s+to\s+slack\s*$', '', message, flags=re.IGNORECASE)
                        message = re.sub(r'\s+on\s+slack\s*$', '', message, flags=re.IGNORECASE)
                        if message:
                            break

            if not message:
                # Fallback: use instruction as message
                message = instruction

            return (channel, message, {})

        elif agent_lower == 'notion':
            # Try to extract page title and content
            import re

            # Pattern: "create page ... titled/named ..."
            title_match = re.search(r'titled?[:\s]+"([^"]+)"', instruction, re.IGNORECASE)
            if not title_match:
                title_match = re.search(r'named?[:\s]+"([^"]+)"', instruction, re.IGNORECASE)

            title = title_match.group(1) if title_match else "New Page"

            # Content is harder to extract, use full instruction
            content = instruction

            return (title, content, {})

        return None

    def register_agent(self, agent_name: str, agent_instance: Any):
        """Register an agent instance for accessing prefetched metadata"""
        self.agent_instances[agent_name] = agent_instance

    def _channel_exists(self, channel_name: str, agent_instance: Any) -> bool:
        """Check if a channel exists in prefetched data"""
        try:
            if not agent_instance or not hasattr(agent_instance, 'metadata_cache'):
                return True  # Can't verify, assume it exists

            metadata = agent_instance.metadata_cache
            if not isinstance(metadata, dict):
                return True

            channels = metadata.get('channels', {})
            if not channels:
                return True

            search_name = channel_name.lstrip('#').lower()

            # channels is a dict keyed by channel ID, iterate over values
            for ch in channels.values():
                if isinstance(ch, dict):
                    ch_name = ch.get('name', '').lower()
                    if ch_name == search_name:
                        return True

            return False

        except Exception:
            return True  # On error, assume exists

    def _get_available_channels(self, agent_instance: Any) -> List[str]:
        """Get list of available channel names"""
        try:
            if not agent_instance or not hasattr(agent_instance, 'metadata_cache'):
                return []

            metadata = agent_instance.metadata_cache
            if not isinstance(metadata, dict):
                return []

            channels = metadata.get('channels', {})
            if not channels:
                return []

            names = []
            # channels is a dict keyed by channel ID, iterate over values
            for ch in channels.values():
                if isinstance(ch, dict):
                    ch_name = ch.get('name', '')
                    if ch_name:
                        names.append(f"#{ch_name}")

            return sorted(names)

        except Exception:
            return []

    def _format_channel_not_found(self, channel_name: str, available_channels: List[str]) -> str:
        """Format message for channel not found"""
        msg = f"\n{'='*70}\n"
        msg += f"⚠️  Channel '#{channel_name}' not found\n"
        msg += f"{'='*70}\n\n"
        msg += f"Available channels:\n"

        # Show channels in columns
        for i, ch in enumerate(available_channels[:20], 1):  # Show max 20
            msg += f"  {ch}"
            if i % 3 == 0:
                msg += "\n"
            else:
                msg += "  "

        if len(available_channels) > 20:
            msg += f"\n  ... and {len(available_channels) - 20} more"

        msg += f"\n\n"
        return msg

    def _calculate_channel_similarity(self, search_name: str, channel_name: str) -> float:
        """
        Calculate similarity score between search term and channel name.
        Returns score from 0.0 (no match) to 1.0 (perfect match).
        """
        search_lower = search_name.lower().replace(' ', '').replace('-', '').replace('_', '')
        channel_lower = channel_name.lower().replace(' ', '').replace('-', '').replace('_', '')

        # Exact match after normalization
        if search_lower == channel_lower:
            return 1.0

        # Contains match
        if search_lower in channel_lower:
            return 0.9

        if channel_lower in search_lower:
            return 0.85

        # Check if all words in search appear in channel
        search_words = search_name.lower().replace('-', ' ').replace('_', ' ').split()
        channel_lower_full = channel_name.lower()

        matches = sum(1 for word in search_words if word in channel_lower_full)
        if matches > 0 and len(search_words) > 0:
            word_match_score = (matches / len(search_words)) * 0.8
            return word_match_score

        # Character-level similarity (simple Levenshtein-like)
        common_chars = sum(1 for c in search_lower if c in channel_lower)
        if len(search_lower) > 0:
            char_similarity = (common_chars / len(search_lower)) * 0.5
            return char_similarity

        return 0.0

    def _resolve_slack_channel(self, channel_name: str, agent_instance: Any) -> str:
        """Resolve fuzzy channel name to actual channel using intelligent matching"""
        try:
            if not agent_instance or not hasattr(agent_instance, 'metadata_cache'):
                return channel_name

            metadata = agent_instance.metadata_cache
            if not isinstance(metadata, dict):
                return channel_name

            channels = metadata.get('channels', {})
            if not channels or not isinstance(channels, dict):
                return channel_name

            # Remove # if present
            search_name = channel_name.lstrip('#')

            # Exact match first (case-insensitive)
            for ch in channels.values():
                if not isinstance(ch, dict):
                    continue
                ch_name = ch.get('name', '')
                if ch_name and ch_name.lower() == search_name.lower():
                    return ch_name

            # Find best fuzzy match using similarity scoring
            best_match = None
            best_score = 0.7  # Minimum threshold for auto-matching

            for ch in channels.values():
                if not isinstance(ch, dict):
                    continue

                ch_name = ch.get('name', '')
                if not ch_name:
                    continue

                score = self._calculate_channel_similarity(search_name, ch_name)

                if score > best_score:
                    best_score = score
                    best_match = ch_name

            if best_match:
                if self.verbose:
                    print(f"[CONFIRM] Fuzzy matched '{channel_name}' → '{best_match}' (score: {best_score:.2f})")
                return best_match

            # No match found, return original
            return channel_name

        except Exception as e:
            # If any error during resolution, just return original
            if self.verbose:
                print(f"[CONFIRM] Channel resolution error: {e}")
            return channel_name

    def confirm_before_execution(
        self,
        agent_name: str,
        instruction: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Show confirmation prompt before execution.

        Args:
            agent_name: Name of the agent
            instruction: The instruction to execute

        Returns:
            (should_execute, modified_instruction)
        """
        agent_lower = agent_name.lower()

        # Extract message content
        extracted = self.extract_message_content(agent_name, instruction)
        if not extracted:
            # Couldn't extract, show generic confirmation
            return self._generic_confirmation(agent_name, instruction)

        destination, content, metadata = extracted

        # Resolve and VALIDATE channel for Slack BEFORE showing confirmation
        if agent_lower == 'slack':
            agent_instance = self.agent_instances.get(agent_name)
            if agent_instance:
                # Try to resolve the channel
                resolved_channel = self._resolve_slack_channel(destination, agent_instance)

                # If no match found, show available channels and let user pick
                if resolved_channel == destination and not self._channel_exists(destination, agent_instance):
                    # Channel doesn't exist - show available options
                    available = self._get_available_channels(agent_instance)

                    if available:
                        # Calculate similarity scores for all channels and show top matches
                        channel_scores = []
                        for ch in available:
                            ch_clean = ch.lstrip('#')
                            score = self._calculate_channel_similarity(destination, ch_clean)
                            channel_scores.append((ch, score))

                        # Sort by similarity score (best matches first)
                        channel_scores.sort(key=lambda x: x[1], reverse=True)
                        sorted_channels = [ch for ch, score in channel_scores]

                        # Use interactive selection if inquirer is available
                        if INQUIRER_AVAILABLE:
                            try:
                                print(f"\n⚠️  Channel '#{destination}' not found.")
                                print(f"📋 Please select the correct channel:\n")

                                # Add cancel option
                                choices = sorted_channels + ['❌ Cancel']

                                questions = [
                                    inquirer.List('channel',
                                                message="Select a channel",
                                                choices=choices,
                                                carousel=True)
                                ]

                                answers = inquirer.prompt(questions, theme=GreenPassion())

                                if not answers or answers['channel'] == '❌ Cancel':
                                    print("❌ Operation cancelled")
                                    return False, None

                                resolved_channel = answers['channel'].lstrip('#')

                            except Exception as e:
                                # Fallback to text input if inquirer fails
                                if self.verbose:
                                    print(f"[CONFIRM] Interactive prompt failed: {e}")
                                print(f"\n{self._format_channel_not_found(destination, sorted_channels[:10])}")
                                choice = input("Enter channel name (or 'c' to cancel): ").strip()

                                if choice.lower() == 'c':
                                    return False, None

                                resolved_channel = choice.lstrip('#')

                                if not self._channel_exists(resolved_channel, agent_instance):
                                    print(f"⚠️  Channel '{resolved_channel}' not found. Operation cancelled.")
                                    return False, None
                        else:
                            # Fallback to text input if inquirer not available
                            print(f"\n{self._format_channel_not_found(destination, sorted_channels[:10])}")
                            choice = input("Enter channel name (or 'c' to cancel): ").strip()

                            if choice.lower() == 'c':
                                return False, None

                            resolved_channel = choice.lstrip('#')

                            if not self._channel_exists(resolved_channel, agent_instance):
                                print(f"⚠️  Channel '{resolved_channel}' not found. Operation cancelled.")
                                return False, None

                # Always rebuild instruction with resolved channel to ensure agent gets correct destination
                destination = resolved_channel
                # Build a clean instruction for the agent with the correct channel
                instruction = f"send '{content}' to #{resolved_channel}"
                if self.verbose:
                    print(f"[CONFIRM] Rebuilt instruction for agent: {instruction}")

        # Show confirmation based on agent type
        if agent_lower == 'slack':
            decision, modified = self.confirmer.confirm_slack_message(
                channel=destination,
                message=content,
                metadata=metadata
            )

        elif agent_lower == 'notion':
            operation = "Create Page" if "create" in instruction.lower() else "Update Page"
            decision, modified = self.confirmer.confirm_notion_operation(
                operation_type=operation,
                page_title=destination,
                content=content,
                metadata=metadata
            )

        else:
            return self._generic_confirmation(agent_name, instruction)

        # Process decision
        if decision == ConfirmationDecision.APPROVED:
            # Rebuild instruction with modified content if changed
            if modified and modified != content:
                modified_instruction = self._rebuild_instruction(
                    agent_name, instruction, content, modified
                )
                return True, modified_instruction
            return True, instruction

        elif decision == ConfirmationDecision.EDIT_AI:
            # User wants AI to modify - return modification request
            # The orchestrator should handle this by asking the agent to revise
            return False, f"[AI_MODIFICATION_REQUESTED] {modified}"

        else:
            # Rejected
            return False, None

    def _rebuild_instruction(
        self,
        agent_name: str,
        original_instruction: str,
        old_content: str,
        new_content: str
    ) -> str:
        """Rebuild instruction with modified content"""
        # Simple replacement
        return original_instruction.replace(old_content, new_content)

    def _generic_confirmation(
        self,
        agent_name: str,
        instruction: str
    ) -> Tuple[bool, Optional[str]]:
        """Generic confirmation when content can't be extracted"""
        print(f"\n{'='*70}")
        print(f"⚠️ **{agent_name.upper()} OPERATION REQUIRES CONFIRMATION**")
        print(f"{'='*70}")
        print(f"\n**Instruction:**")
        print(instruction)
        print()

        try:
            choice = input("Approve this operation? [y/n]: ").strip().lower()
            if choice == 'y':
                print("✅ Approved!")
                return True, instruction
            else:
                print("❌ Rejected!")
                return False, None
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Cancelled")
            return False, None
