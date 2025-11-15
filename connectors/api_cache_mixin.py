"""
API Cache Mixin for Agents

Provides caching capabilities for agent API calls to reduce latency and costs.
"""

from typing import Any, Dict, Optional, Callable
import functools
import inspect


class APICacheMixin:
    """
    Mixin class that adds API caching capabilities to agents.

    Usage:
        class MyAgent(APICacheMixin, BaseAgent):
            def __init__(self, ...):
                super().__init__(...)
                self.service_name = "jira"  # Set service name for cache namespacing

            @cache_api_call(endpoint="list_issues", ttl=180)
            async def list_issues(self, project_key):
                # Your API call logic
                ...
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_cache = None  # Will be injected by orchestrator
        self.service_name = "unknown"  # Should be overridden by subclass

    def set_api_cache(self, api_cache: Any):
        """Set the API cache instance (called by orchestrator)"""
        self.api_cache = api_cache

    def get_cached_api_response(
        self,
        endpoint: str,
        params: Dict,
        compute_fn: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get API response from cache or compute if missing.

        Args:
            endpoint: API endpoint name (e.g., "list_issues")
            params: Parameters for the API call
            compute_fn: Function to call if cache miss
            ttl: Optional TTL override in seconds

        Returns:
            API response (from cache or fresh call)
        """
        # If no cache available, just compute
        if self.api_cache is None:
            return compute_fn()

        # Try to get from cache
        cached = self.api_cache.get(self.service_name, endpoint, params)
        if cached is not None:
            return cached

        # Cache miss - compute result
        result = compute_fn()

        # Store in cache
        self.api_cache.set(self.service_name, endpoint, params, result)

        return result

    def invalidate_api_cache(self, endpoint: Optional[str] = None):
        """
        Invalidate API cache for this service.

        Args:
            endpoint: If specified, only invalidate this endpoint. Otherwise invalidate all.
        """
        if self.api_cache is None:
            return

        if endpoint:
            self.api_cache.invalidate_endpoint(self.service_name, endpoint)
        else:
            self.api_cache.invalidate_service(self.service_name)


def cache_api_call(endpoint: str, ttl: Optional[int] = None, invalidate_on_write: bool = False):
    """
    Decorator to automatically cache API calls.

    Args:
        endpoint: Name of the API endpoint
        ttl: Time to live in seconds (optional, uses service default if not specified)
        invalidate_on_write: If True, invalidate cache after successful execution
                             (for write operations like create/update/delete)

    Example:
        @cache_api_call(endpoint="get_issue", ttl=300)
        async def get_issue(self, issue_key: str):
            # API call logic
            ...

        @cache_api_call(endpoint="create_issue", invalidate_on_write=True)
        async def create_issue(self, data: Dict):
            # Write operation - cache will be invalidated after
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Check if instance has API cache mixin
            if not hasattr(self, 'api_cache') or self.api_cache is None:
                # No cache - just call function
                return await func(self, *args, **kwargs)

            # Build cache key from function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            # Create params dict from arguments (excluding self)
            params = {
                k: str(v) for k, v in bound_args.arguments.items() if k != 'self'
            }

            # For write operations, invalidate cache
            if invalidate_on_write:
                # Execute the function
                result = await func(self, *args, **kwargs)

                # Invalidate cache after successful write
                self.invalidate_api_cache()

                return result

            # For read operations, use cache
            service_name = getattr(self, 'service_name', 'unknown')

            # Check cache
            cached = self.api_cache.get(service_name, endpoint, params)
            if cached is not None:
                if getattr(self, 'verbose', False):
                    print(f"[{service_name.upper()} CACHE] Hit: {endpoint}")
                return cached

            # Cache miss - call function
            if getattr(self, 'verbose', False):
                print(f"[{service_name.upper()} CACHE] Miss: {endpoint}")

            result = await func(self, *args, **kwargs)

            # Cache the result
            self.api_cache.set(service_name, endpoint, params, result)

            return result

        return wrapper
    return decorator


def create_cached_agent_wrapper(agent_instance: Any, api_cache: Any) -> Any:
    """
    Wrap an existing agent instance with API caching.

    This is a utility function for agents that don't use the mixin.

    Args:
        agent_instance: The agent to wrap
        api_cache: APIResponseCache instance

    Returns:
        Agent instance with caching enabled
    """
    if hasattr(agent_instance, 'set_api_cache'):
        agent_instance.set_api_cache(api_cache)

    return agent_instance
