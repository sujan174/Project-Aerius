"""
Circuit Breaker Pattern Implementation

Prevents cascading failures by automatically disabling agents that are
repeatedly failing and allowing them to recover after a timeout period.

Circuit States:
- CLOSED: Normal operation, requests allowed
- OPEN: Too many failures, requests blocked
- HALF_OPEN: Testing recovery, allowing one request

Features:
- Automatic failure detection (N consecutive failures → OPEN)
- Automatic recovery attempts after timeout
- Graceful degradation (fast failure instead of slow timeouts)
- Per-agent circuit tracking
- Detailed metrics and logging
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Callable, Any
import asyncio


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit open, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitMetrics:
    """Metrics for a circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Rejected due to circuit being open
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: Optional[float] = None
    circuit_opened_count: int = 0
    circuit_half_opened_count: int = 0

    def success_rate(self) -> float:
        """Calculate success rate (0.0-1.0)"""
        total = self.successful_requests + self.failed_requests
        if total == 0:
            return 1.0
        return self.successful_requests / total


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 300.0  # 5 minutes before attempting recovery
    half_open_timeout: float = 10.0  # Timeout for half-open test requests


class CircuitBreakerError(Exception):
    """Raised when circuit breaker blocks a request"""
    def __init__(self, agent_name: str, message: str):
        self.agent_name = agent_name
        super().__init__(message)


class CircuitBreaker:
    """
    Circuit breaker for agent health management.

    Automatically disables agents after repeated failures and attempts
    recovery after a timeout period.
    """

    def __init__(self, config: Optional[CircuitConfig] = None, verbose: bool = False):
        self.config = config or CircuitConfig()
        self.verbose = verbose

        # Per-agent circuit state
        self.circuits: Dict[str, CircuitState] = {}
        self.metrics: Dict[str, CircuitMetrics] = {}

        # Locks for thread-safe updates
        self._locks: Dict[str, asyncio.Lock] = {}

    async def _get_lock(self, agent_name: str) -> asyncio.Lock:
        """Get or create lock for an agent"""
        if agent_name not in self._locks:
            self._locks[agent_name] = asyncio.Lock()
        return self._locks[agent_name]

    def _get_circuit_state(self, agent_name: str) -> CircuitState:
        """Get current state of circuit for agent"""
        return self.circuits.get(agent_name, CircuitState.CLOSED)

    def _get_metrics(self, agent_name: str) -> CircuitMetrics:
        """Get or create metrics for agent"""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = CircuitMetrics()
        return self.metrics[agent_name]

    async def can_execute(self, agent_name: str) -> tuple[bool, Optional[str]]:
        """
        Check if request should be allowed to proceed.

        Returns:
            (allowed, reason): Boolean indicating if allowed, and reason if not
        """
        lock = await self._get_lock(agent_name)
        async with lock:
            state = self._get_circuit_state(agent_name)
            metrics = self._get_metrics(agent_name)

            if state == CircuitState.CLOSED:
                # Normal operation
                return (True, None)

            elif state == CircuitState.OPEN:
                # Check if timeout has elapsed
                if metrics.last_failure_time is None:
                    # Shouldn't happen, but handle gracefully
                    return (True, None)

                elapsed = time.time() - metrics.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    # Timeout elapsed, try recovery
                    await self._transition_to_half_open(agent_name)
                    return (True, "Circuit transitioning to HALF_OPEN for recovery test")

                # Still within timeout, reject request
                metrics.rejected_requests += 1
                remaining = self.config.timeout_seconds - elapsed
                return (False, f"Circuit OPEN for {agent_name}. Recovery attempt in {remaining:.0f}s")

            elif state == CircuitState.HALF_OPEN:
                # Allow one request to test recovery
                return (True, "Circuit in HALF_OPEN state, testing recovery")

        return (False, "Unknown circuit state")

    async def record_success(self, agent_name: str):
        """Record successful request"""
        lock = await self._get_lock(agent_name)
        async with lock:
            metrics = self._get_metrics(agent_name)
            state = self._get_circuit_state(agent_name)

            metrics.total_requests += 1
            metrics.successful_requests += 1
            metrics.consecutive_successes += 1
            metrics.consecutive_failures = 0
            metrics.last_success_time = time.time()

            if state == CircuitState.HALF_OPEN:
                # Success in half-open state
                if metrics.consecutive_successes >= self.config.success_threshold:
                    # Enough successes, close the circuit
                    await self._transition_to_closed(agent_name)
                    if self.verbose:
                        print(f"🟢 Circuit CLOSED for {agent_name} - Service recovered")

            elif state == CircuitState.OPEN:
                # Shouldn't happen (request should have been rejected), but handle it
                await self._transition_to_half_open(agent_name)

    async def record_failure(self, agent_name: str, error: Optional[Exception] = None):
        """Record failed request"""
        lock = await self._get_lock(agent_name)
        async with lock:
            metrics = self._get_metrics(agent_name)
            state = self._get_circuit_state(agent_name)

            metrics.total_requests += 1
            metrics.failed_requests += 1
            metrics.consecutive_failures += 1
            metrics.consecutive_successes = 0
            metrics.last_failure_time = time.time()

            if state == CircuitState.CLOSED:
                # Check if threshold exceeded
                if metrics.consecutive_failures >= self.config.failure_threshold:
                    await self._transition_to_open(agent_name)
                    if self.verbose:
                        print(f"🔴 Circuit OPENED for {agent_name} after {metrics.consecutive_failures} consecutive failures")
                        print(f"   Recovery attempt in {self.config.timeout_seconds}s")

            elif state == CircuitState.HALF_OPEN:
                # Failure during recovery test, back to open
                await self._transition_to_open(agent_name)
                if self.verbose:
                    print(f"🟡 Circuit back to OPEN for {agent_name} - Recovery test failed")

    async def _transition_to_open(self, agent_name: str):
        """Transition circuit to OPEN state"""
        metrics = self._get_metrics(agent_name)
        self.circuits[agent_name] = CircuitState.OPEN
        metrics.last_state_change = time.time()
        metrics.circuit_opened_count += 1

    async def _transition_to_half_open(self, agent_name: str):
        """Transition circuit to HALF_OPEN state"""
        metrics = self._get_metrics(agent_name)
        self.circuits[agent_name] = CircuitState.HALF_OPEN
        metrics.last_state_change = time.time()
        metrics.circuit_half_opened_count += 1
        metrics.consecutive_successes = 0  # Reset for recovery test

    async def _transition_to_closed(self, agent_name: str):
        """Transition circuit to CLOSED state"""
        metrics = self._get_metrics(agent_name)
        self.circuits[agent_name] = CircuitState.CLOSED
        metrics.last_state_change = time.time()
        metrics.consecutive_failures = 0  # Reset failure count

    async def execute_with_circuit_breaker(
        self,
        agent_name: str,
        operation: Callable[[], Any]
    ) -> Any:
        """
        Execute operation with circuit breaker protection.

        Args:
            agent_name: Name of the agent
            operation: Async callable to execute

        Returns:
            Result of operation

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from operation
        """
        # Check if allowed
        allowed, reason = await self.can_execute(agent_name)
        if not allowed:
            raise CircuitBreakerError(agent_name, reason)

        # Execute operation
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation()
            else:
                result = operation()

            # Record success
            await self.record_success(agent_name)
            return result

        except Exception as e:
            # Record failure
            await self.record_failure(agent_name, e)
            raise

    def get_circuit_state(self, agent_name: str) -> CircuitState:
        """Get current circuit state for agent"""
        return self._get_circuit_state(agent_name)

    def get_metrics(self, agent_name: str) -> CircuitMetrics:
        """Get metrics for agent"""
        return self._get_metrics(agent_name)

    def get_all_states(self) -> Dict[str, CircuitState]:
        """Get circuit states for all agents"""
        return self.circuits.copy()

    def get_all_metrics(self) -> Dict[str, CircuitMetrics]:
        """Get metrics for all agents"""
        return self.metrics.copy()

    async def reset_circuit(self, agent_name: str):
        """Manually reset circuit to CLOSED state"""
        lock = await self._get_lock(agent_name)
        async with lock:
            self.circuits[agent_name] = CircuitState.CLOSED
            metrics = self._get_metrics(agent_name)
            metrics.consecutive_failures = 0
            metrics.consecutive_successes = 0
            metrics.last_state_change = time.time()

            if self.verbose:
                print(f"🔄 Circuit manually reset to CLOSED for {agent_name}")

    def get_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health summary for all agents.

        Returns dict with agent_name → {state, success_rate, failures, last_failure}
        """
        summary = {}
        for agent_name in self.metrics.keys():
            metrics = self.metrics[agent_name]
            state = self.circuits.get(agent_name, CircuitState.CLOSED)

            summary[agent_name] = {
                'state': state.value,
                'success_rate': metrics.success_rate(),
                'total_requests': metrics.total_requests,
                'failed_requests': metrics.failed_requests,
                'rejected_requests': metrics.rejected_requests,
                'consecutive_failures': metrics.consecutive_failures,
                'last_failure_time': metrics.last_failure_time,
                'last_success_time': metrics.last_success_time,
                'circuit_opened_count': metrics.circuit_opened_count
            }

        return summary
