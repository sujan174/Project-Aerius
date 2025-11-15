"""
Advanced Caching System with Semantic Deduplication, Persistence, and Warming

Features:
1. Semantic Cache - Matches similar queries using embeddings
2. Persistent Cache - Saves cache to disk for cross-session usage
3. Cache Warming - Preloads common patterns on startup
4. API Response Cache - Caches external API responses (Jira, GitHub, etc.)
"""

import os
import json
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
import hashlib
import numpy as np


@dataclass
class SemanticCacheEntry:
    """Cache entry with semantic embeddings for similarity matching"""
    key: str
    query: str  # Original query text
    embedding: Optional[List[float]]  # Semantic embedding
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[float] = None
    similarity_threshold: float = 0.85  # For semantic matching

    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def touch(self):
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict:
        """Serialize for persistence"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticCacheEntry':
        """Deserialize from persistence"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        return cls(**data)


class SemanticCache:
    """
    Semantic cache that matches similar queries using embeddings.

    Example:
        "show my tickets" ≈ "list my tasks" ≈ "get my issues"
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        embedding_model: Optional[Callable[[str], List[float]]] = None,
        verbose: bool = False
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.verbose = verbose
        self._entries: List[SemanticCacheEntry] = []
        self._lock = threading.RLock()
        self.semantic_hits = 0
        self.semantic_misses = 0

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text. Returns None if no embedding model."""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model(text)
        except Exception as e:
            if self.verbose:
                print(f"[SEMANTIC CACHE] Embedding error: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def find_similar(self, query: str, threshold: Optional[float] = None) -> Optional[SemanticCacheEntry]:
        """Find semantically similar cached entry"""
        threshold = threshold or self.similarity_threshold
        embedding = self._get_embedding(query)

        if embedding is None:
            return None

        with self._lock:
            best_match = None
            best_similarity = threshold

            for entry in self._entries:
                if entry.is_expired():
                    continue

                if entry.embedding is None:
                    continue

                similarity = self._cosine_similarity(embedding, entry.embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry

            if best_match:
                self.semantic_hits += 1
                if self.verbose:
                    print(f"[SEMANTIC CACHE] Found similar query: '{best_match.query}' (similarity: {best_similarity:.3f})")
                best_match.touch()
                return best_match
            else:
                self.semantic_misses += 1
                return None

    def add(self, key: str, query: str, value: Any, ttl_seconds: Optional[float] = None):
        """Add entry with semantic embedding"""
        embedding = self._get_embedding(query)

        entry = SemanticCacheEntry(
            key=key,
            query=query,
            embedding=embedding,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds,
            similarity_threshold=self.similarity_threshold
        )

        with self._lock:
            self._entries.append(entry)

            if self.verbose:
                print(f"[SEMANTIC CACHE] Added: '{query[:50]}...'")

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            original_count = len(self._entries)
            self._entries = [e for e in self._entries if not e.is_expired()]
            removed = original_count - len(self._entries)

            if self.verbose and removed > 0:
                print(f"[SEMANTIC CACHE] Cleaned up {removed} expired entries")

            return removed

    def get_stats(self) -> Dict[str, Any]:
        """Get semantic cache statistics"""
        with self._lock:
            total = self.semantic_hits + self.semantic_misses
            hit_rate = self.semantic_hits / total if total > 0 else 0.0

            return {
                'entries': len(self._entries),
                'semantic_hits': self.semantic_hits,
                'semantic_misses': self.semantic_misses,
                'semantic_hit_rate': hit_rate
            }


class PersistentCache:
    """
    Persistent cache that saves to disk and survives restarts.

    Features:
    - Automatic periodic saves
    - Load on startup
    - JSON-based for readability
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        cache_file: str = "aerius_cache.json",
        auto_save_interval: int = 300,  # Save every 5 minutes
        verbose: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / cache_file
        self.auto_save_interval = auto_save_interval
        self.verbose = verbose

        self._cache: Dict[str, Dict] = {}
        self._lock = threading.RLock()
        self._last_save = datetime.now()
        self._dirty = False

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        self.load()

        # Start auto-save thread
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()

    def load(self) -> int:
        """Load cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)

                with self._lock:
                    self._cache = data
                    loaded_count = len(data)

                if self.verbose:
                    print(f"[PERSISTENT CACHE] Loaded {loaded_count} entries from {self.cache_file}")

                return loaded_count
            else:
                if self.verbose:
                    print(f"[PERSISTENT CACHE] No existing cache file found")
                return 0

        except Exception as e:
            if self.verbose:
                print(f"[PERSISTENT CACHE] Load error: {e}")
            return 0

    def save(self) -> bool:
        """Save cache to disk"""
        try:
            with self._lock:
                # Create temporary file first
                temp_file = self.cache_file.with_suffix('.tmp')

                with open(temp_file, 'w') as f:
                    json.dump(self._cache, f, indent=2, default=str)

                # Atomic rename
                temp_file.replace(self.cache_file)

                self._last_save = datetime.now()
                self._dirty = False

                if self.verbose:
                    print(f"[PERSISTENT CACHE] Saved {len(self._cache)} entries to {self.cache_file}")

                return True

        except Exception as e:
            if self.verbose:
                print(f"[PERSISTENT CACHE] Save error: {e}")
            return False

    def _auto_save_loop(self):
        """Background thread for automatic saves"""
        while True:
            time.sleep(self.auto_save_interval)

            if self._dirty:
                self.save()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check expiration
                if 'ttl_seconds' in entry and entry['ttl_seconds'] is not None:
                    created_at = datetime.fromisoformat(entry['created_at'])
                    age = (datetime.now() - created_at).total_seconds()

                    if age > entry['ttl_seconds']:
                        # Expired
                        del self._cache[key]
                        self._dirty = True
                        return None

                # Update access stats
                entry['last_accessed'] = datetime.now().isoformat()
                entry['access_count'] = entry.get('access_count', 0) + 1
                self._dirty = True

                return entry['value']

            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """Set value in cache"""
        with self._lock:
            entry = {
                'key': key,
                'value': value,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0,
                'ttl_seconds': ttl_seconds
            }

            self._cache[key] = entry
            self._dirty = True

    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            now = datetime.now()
            expired_keys = []

            for key, entry in self._cache.items():
                if 'ttl_seconds' in entry and entry['ttl_seconds'] is not None:
                    created_at = datetime.fromisoformat(entry['created_at'])
                    age = (now - created_at).total_seconds()

                    if age > entry['ttl_seconds']:
                        expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            if expired_keys:
                self._dirty = True

                if self.verbose:
                    print(f"[PERSISTENT CACHE] Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'last_save': self._last_save.isoformat(),
                'dirty': self._dirty
            }


class CacheWarmer:
    """
    Preloads common query patterns into cache on startup.

    Patterns are learned from historical usage and manually configured.
    """

    def __init__(
        self,
        cache: Any,  # Can be IntelligentCache or HybridCache
        patterns_file: str = ".cache/common_patterns.json",
        verbose: bool = False
    ):
        self.cache = cache
        self.patterns_file = Path(patterns_file)
        self.verbose = verbose

        # Default common patterns
        self.default_patterns = {
            "jira_patterns": [
                "show my jira tickets",
                "list my tasks",
                "get my issues",
                "show open tickets",
                "list assigned to me"
            ],
            "github_patterns": [
                "show my pull requests",
                "list my PRs",
                "show open issues",
                "list repositories"
            ],
            "slack_patterns": [
                "show recent messages",
                "list channels",
                "get notifications"
            ]
        }

    def load_patterns(self) -> Dict[str, List[str]]:
        """Load patterns from file or use defaults"""
        try:
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    patterns = json.load(f)

                if self.verbose:
                    print(f"[CACHE WARMER] Loaded patterns from {self.patterns_file}")

                return patterns
            else:
                if self.verbose:
                    print(f"[CACHE WARMER] Using default patterns")

                return self.default_patterns

        except Exception as e:
            if self.verbose:
                print(f"[CACHE WARMER] Error loading patterns: {e}")

            return self.default_patterns

    def warm_cache(self, compute_fn: Callable[[str], Any]):
        """
        Warm cache with common patterns.

        Args:
            compute_fn: Function that computes the result for a pattern
        """
        patterns = self.load_patterns()
        warmed_count = 0

        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                try:
                    # Only warm if not already cached
                    cache_key = f"intent:{hashlib.md5(pattern.encode()).hexdigest()[:16]}"

                    if hasattr(self.cache, 'get'):
                        existing = self.cache.get(cache_key)
                        if existing is not None:
                            continue

                    # Compute and cache
                    result = compute_fn(pattern)

                    if hasattr(self.cache, 'set'):
                        self.cache.set(cache_key, result, ttl_seconds=3600)  # 1 hour TTL

                    warmed_count += 1

                    if self.verbose:
                        print(f"[CACHE WARMER] Warmed pattern: '{pattern}'")

                except Exception as e:
                    if self.verbose:
                        print(f"[CACHE WARMER] Error warming '{pattern}': {e}")

        if self.verbose:
            print(f"[CACHE WARMER] Warmed {warmed_count} patterns")

        return warmed_count


class APIResponseCache:
    """
    Caches API responses from external services (Jira, GitHub, Slack, etc.)

    Features:
    - Per-endpoint caching with configurable TTL
    - Request deduplication
    - Automatic invalidation on write operations
    """

    def __init__(
        self,
        default_ttl: int = 300,  # 5 minutes
        max_size: int = 5000,
        verbose: bool = False
    ):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.verbose = verbose

        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.RLock()

        # Per-service TTL configurations
        self.service_ttls = {
            'jira': {
                'list_issues': 180,      # 3 minutes - frequently changing
                'get_issue': 300,         # 5 minutes
                'search_issues': 180,     # 3 minutes
                'get_user': 3600,         # 1 hour - rarely changes
                'list_projects': 1800,    # 30 minutes
            },
            'github': {
                'list_prs': 120,          # 2 minutes - frequently changing
                'get_pr': 300,            # 5 minutes
                'list_issues': 180,       # 3 minutes
                'get_repo': 1800,         # 30 minutes
                'list_repos': 600,        # 10 minutes
            },
            'slack': {
                'list_messages': 60,      # 1 minute - real-time
                'get_user': 3600,         # 1 hour
                'list_channels': 1800,    # 30 minutes
            }
        }

        self.hits = 0
        self.misses = 0

    def _make_key(self, service: str, endpoint: str, params: Dict) -> str:
        """Create cache key from service, endpoint, and parameters"""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:16]
        return f"api:{service}:{endpoint}:{params_hash}"

    def _get_ttl(self, service: str, endpoint: str) -> int:
        """Get TTL for specific service/endpoint combination"""
        if service in self.service_ttls:
            return self.service_ttls[service].get(endpoint, self.default_ttl)
        return self.default_ttl

    def get(self, service: str, endpoint: str, params: Dict) -> Optional[Any]:
        """Get cached API response"""
        key = self._make_key(service, endpoint, params)

        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            created_at = datetime.fromisoformat(entry['created_at'])
            age = (datetime.now() - created_at).total_seconds()

            if age > entry['ttl']:
                del self._cache[key]
                self.misses += 1

                if self.verbose:
                    print(f"[API CACHE] Expired: {service}.{endpoint}")

                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)

            self.hits += 1

            if self.verbose:
                print(f"[API CACHE] Hit: {service}.{endpoint} (age: {age:.1f}s)")

            return entry['response']

    def set(self, service: str, endpoint: str, params: Dict, response: Any):
        """Cache API response"""
        key = self._make_key(service, endpoint, params)
        ttl = self._get_ttl(service, endpoint)

        with self._lock:
            entry = {
                'service': service,
                'endpoint': endpoint,
                'params': params,
                'response': response,
                'created_at': datetime.now().isoformat(),
                'ttl': ttl
            }

            if key in self._cache:
                del self._cache[key]

            self._cache[key] = entry

            # Evict oldest if over size limit
            if len(self._cache) > self.max_size:
                oldest_key, _ = self._cache.popitem(last=False)

                if self.verbose:
                    print(f"[API CACHE] Evicted oldest entry")

            if self.verbose:
                print(f"[API CACHE] Cached: {service}.{endpoint} (TTL: {ttl}s)")

    def invalidate_service(self, service: str) -> int:
        """Invalidate all cache entries for a service"""
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry['service'] == service
            ]

            for key in keys_to_remove:
                del self._cache[key]

            if self.verbose and keys_to_remove:
                print(f"[API CACHE] Invalidated {len(keys_to_remove)} entries for {service}")

            return len(keys_to_remove)

    def invalidate_endpoint(self, service: str, endpoint: str) -> int:
        """Invalidate cache for specific endpoint"""
        with self._lock:
            keys_to_remove = [
                key for key, entry in self._cache.items()
                if entry['service'] == service and entry['endpoint'] == endpoint
            ]

            for key in keys_to_remove:
                del self._cache[key]

            if self.verbose and keys_to_remove:
                print(f"[API CACHE] Invalidated {len(keys_to_remove)} entries for {service}.{endpoint}")

            return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0

            # Group by service
            by_service = {}
            for entry in self._cache.values():
                service = entry['service']
                by_service[service] = by_service.get(service, 0) + 1

            return {
                'size': len(self._cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'by_service': by_service
            }


class HybridCache:
    """
    Unified cache combining:
    - Exact match caching (fast)
    - Semantic caching (similar queries)
    - Persistent storage (survives restarts)
    - API response caching (external services)
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        enable_semantic: bool = True,
        enable_persistent: bool = True,
        enable_api_cache: bool = True,
        embedding_model: Optional[Callable[[str], List[float]]] = None,
        verbose: bool = False
    ):
        self.verbose = verbose

        # Core exact-match cache (from existing IntelligentCache)
        from intelligence.system import IntelligentCache
        self.exact_cache = IntelligentCache(
            max_size=1000,
            default_ttl_seconds=300,
            verbose=verbose
        )

        # Semantic cache for similar queries
        self.semantic_cache = SemanticCache(
            similarity_threshold=0.85,
            embedding_model=embedding_model,
            verbose=verbose
        ) if enable_semantic else None

        # Persistent cache for cross-session storage
        self.persistent_cache = PersistentCache(
            cache_dir=cache_dir,
            verbose=verbose
        ) if enable_persistent else None

        # API response cache
        self.api_cache = APIResponseCache(
            verbose=verbose
        ) if enable_api_cache else None

        # Cache warmer
        self.cache_warmer = CacheWarmer(
            cache=self.exact_cache,
            patterns_file=f"{cache_dir}/common_patterns.json",
            verbose=verbose
        )

    def get(self, key: str, query: Optional[str] = None) -> Optional[Any]:
        """
        Multi-level cache lookup:
        1. Exact match in memory
        2. Semantic match (if query provided)
        3. Persistent cache
        """
        # Level 1: Exact match (fastest)
        result = self.exact_cache.get(key)
        if result is not None:
            return result

        # Level 2: Semantic match (if enabled and query provided)
        if self.semantic_cache and query:
            entry = self.semantic_cache.find_similar(query)
            if entry:
                # Promote to exact cache
                self.exact_cache.set(key, entry.value)
                return entry.value

        # Level 3: Persistent cache
        if self.persistent_cache:
            result = self.persistent_cache.get(key)
            if result is not None:
                # Promote to exact cache
                self.exact_cache.set(key, result)
                return result

        return None

    def set(self, key: str, value: Any, query: Optional[str] = None, ttl_seconds: Optional[float] = None):
        """Set value in all enabled caches"""
        # Set in exact cache
        self.exact_cache.set(key, value, ttl_seconds)

        # Set in semantic cache (if enabled and query provided)
        if self.semantic_cache and query:
            self.semantic_cache.add(key, query, value, ttl_seconds)

        # Set in persistent cache
        if self.persistent_cache:
            self.persistent_cache.set(key, value, ttl_seconds)

    def get_api_response(self, service: str, endpoint: str, params: Dict) -> Optional[Any]:
        """Get cached API response"""
        if self.api_cache:
            return self.api_cache.get(service, endpoint, params)
        return None

    def set_api_response(self, service: str, endpoint: str, params: Dict, response: Any):
        """Cache API response"""
        if self.api_cache:
            self.api_cache.set(service, endpoint, params, response)

    def invalidate_api_service(self, service: str):
        """Invalidate all API cache for a service (e.g., after write operation)"""
        if self.api_cache:
            self.api_cache.invalidate_service(service)

    def warm_cache(self, compute_fn: Callable[[str], Any]) -> int:
        """Warm cache with common patterns"""
        return self.cache_warmer.warm_cache(compute_fn)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'exact_cache': self.exact_cache.get_stats(),
        }

        if self.semantic_cache:
            stats['semantic_cache'] = self.semantic_cache.get_stats()

        if self.persistent_cache:
            stats['persistent_cache'] = self.persistent_cache.get_stats()

        if self.api_cache:
            stats['api_cache'] = self.api_cache.get_stats()

        return stats

    def cleanup(self):
        """Cleanup expired entries across all caches"""
        if self.semantic_cache:
            self.semantic_cache.cleanup_expired()

        if self.persistent_cache:
            self.persistent_cache.cleanup_expired()
            self.persistent_cache.save()
