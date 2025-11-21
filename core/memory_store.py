"""
Semantic Memory Store

Lightweight embedding-based memory for storing and retrieving user instructions/preferences.
Uses cosine similarity for semantic search.

Author: AI System
Version: 1.0
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Memory:
    """A single memory item"""
    id: str
    content: str  # The actual instruction/preference
    category: str  # timezone, default, behavior, etc.
    embedding: Optional[List[float]]  # Vector embedding
    created_at: float
    updated_at: float
    access_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticMemoryStore:
    """
    Embedding-based memory store with semantic search.

    Features:
    - Store memories with embeddings
    - Semantic similarity search
    - Persistent storage
    - LRU-like access tracking
    """

    def __init__(
        self,
        storage_path: str = None,
        embedding_fn=None,
        max_memories: int = 1000,
        verbose: bool = False
    ):
        """
        Initialize memory store.

        Args:
            storage_path: Path to JSON file for persistence
            embedding_fn: Async function to generate embeddings (text -> List[float])
            max_memories: Maximum number of memories to store
            verbose: Enable verbose logging
        """
        self.storage_path = storage_path
        self.embedding_fn = embedding_fn
        self.max_memories = max_memories
        self.verbose = verbose

        # In-memory storage
        self.memories: Dict[str, Memory] = {}

        # Statistics
        self.total_saves = 0
        self.total_retrievals = 0
        self.cache_hits = 0

        # Load from disk
        if storage_path:
            self._load()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    async def save(
        self,
        content: str,
        category: str = "general",
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save a memory with embedding.

        Args:
            content: The memory content
            category: Category (timezone, default, behavior, etc.)
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        self.total_saves += 1

        # Generate ID
        memory_id = self._generate_id(content)

        # Check if already exists (update)
        is_update = memory_id in self.memories

        # Generate embedding if function provided
        embedding = None
        if self.embedding_fn:
            try:
                embedding = await self.embedding_fn(content)
            except Exception as e:
                if self.verbose:
                    print(f"[MEMORY] Embedding generation failed: {e}")

        # Create memory
        now = time.time()
        memory = Memory(
            id=memory_id,
            content=content,
            category=category,
            embedding=embedding,
            created_at=self.memories[memory_id].created_at if is_update else now,
            updated_at=now,
            access_count=self.memories[memory_id].access_count if is_update else 0,
            metadata=metadata or {}
        )

        # Store
        self.memories[memory_id] = memory

        # Evict old memories if over limit
        if len(self.memories) > self.max_memories:
            self._evict_oldest()

        # Persist
        if self.storage_path:
            self._save()

        if self.verbose:
            action = "Updated" if is_update else "Saved"
            print(f"[MEMORY] {action}: {content[:50]}... (category: {category})")

        return memory_id

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str = None,
        threshold: float = 0.3
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieve relevant memories using semantic search.

        Args:
            query: Search query
            top_k: Number of results to return
            category_filter: Filter by category
            threshold: Minimum similarity threshold

        Returns:
            List of (memory, similarity_score) tuples
        """
        self.total_retrievals += 1

        if not self.memories:
            return []

        # Filter by category if specified
        candidates = list(self.memories.values())
        if category_filter:
            candidates = [m for m in candidates if m.category == category_filter]

        if not candidates:
            return []

        # If no embedding function, fall back to keyword matching
        if not self.embedding_fn:
            return self._keyword_search(query, candidates, top_k)

        # Generate query embedding
        try:
            query_embedding = await self.embedding_fn(query)
        except Exception as e:
            if self.verbose:
                print(f"[MEMORY] Query embedding failed: {e}")
            return self._keyword_search(query, candidates, top_k)

        # Calculate similarities
        results = []
        for memory in candidates:
            if memory.embedding:
                similarity = self._cosine_similarity(query_embedding, memory.embedding)
                if similarity >= threshold:
                    results.append((memory, similarity))
                    # Update access count
                    memory.access_count += 1

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _keyword_search(
        self,
        query: str,
        candidates: List[Memory],
        top_k: int
    ) -> List[Tuple[Memory, float]]:
        """Fallback keyword-based search"""
        query_words = set(query.lower().split())

        results = []
        for memory in candidates:
            content_words = set(memory.content.lower().split())
            # Jaccard similarity
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            similarity = intersection / union if union > 0 else 0

            if similarity > 0:
                results.append((memory, similarity))
                memory.access_count += 1

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)

        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get memory by ID"""
        return self.memories.get(memory_id)

    def get_all(self, category: str = None) -> List[Memory]:
        """Get all memories, optionally filtered by category"""
        memories = list(self.memories.values())
        if category:
            memories = [m for m in memories if m.category == category]
        return sorted(memories, key=lambda m: m.updated_at, reverse=True)

    def delete(self, memory_id: str) -> bool:
        """Delete a memory"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            if self.storage_path:
                self._save()
            return True
        return False

    def clear(self, category: str = None):
        """Clear memories, optionally by category"""
        if category:
            self.memories = {
                k: v for k, v in self.memories.items()
                if v.category != category
            }
        else:
            self.memories = {}

        if self.storage_path:
            self._save()

    def _evict_oldest(self):
        """Evict least recently accessed memories"""
        if len(self.memories) <= self.max_memories:
            return

        # Sort by access count and update time
        sorted_memories = sorted(
            self.memories.items(),
            key=lambda x: (x[1].access_count, x[1].updated_at)
        )

        # Remove oldest
        to_remove = len(self.memories) - self.max_memories
        for memory_id, _ in sorted_memories[:to_remove]:
            del self.memories[memory_id]

    def format_for_prompt(self, memories: List[Tuple[Memory, float]] = None) -> str:
        """
        Format memories for injection into system prompt.

        Args:
            memories: List of (memory, score) tuples, or None for all

        Returns:
            Formatted string for prompt injection
        """
        if memories is None:
            # Get all memories
            all_memories = self.get_all()
            if not all_memories:
                return ""
            memories = [(m, 1.0) for m in all_memories]

        if not memories:
            return ""

        # Group by category
        by_category: Dict[str, List[str]] = {}
        for memory, score in memories:
            cat = memory.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(memory.content)

        # Format
        lines = [
            "# User Preferences & Instructions",
            "",
            "The following MUST be applied to all responses:"
        ]

        category_labels = {
            'timezone': 'Time & Timezone',
            'default': 'Default Settings',
            'behavior': 'Behavior Rules',
            'formatting': 'Formatting',
            'notification': 'Notifications',
            'style': 'Communication Style',
            'general': 'General Preferences'
        }

        for cat, items in by_category.items():
            label = category_labels.get(cat, cat.title())
            lines.append(f"\n**{label}:**")
            for item in items:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict:
        """Get memory store statistics"""
        categories = {}
        for memory in self.memories.values():
            cat = memory.category
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_memories': len(self.memories),
            'total_saves': self.total_saves,
            'total_retrievals': self.total_retrievals,
            'categories': categories,
            'has_embeddings': any(m.embedding for m in self.memories.values())
        }

    def _save(self):
        """Save memories to disk"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                'memories': {
                    k: {
                        'id': v.id,
                        'content': v.content,
                        'category': v.category,
                        'embedding': v.embedding,
                        'created_at': v.created_at,
                        'updated_at': v.updated_at,
                        'access_count': v.access_count,
                        'metadata': v.metadata
                    }
                    for k, v in self.memories.items()
                },
                'saved_at': time.time()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            if self.verbose:
                print(f"[MEMORY] Save error: {e}")

    def _load(self):
        """Load memories from disk"""
        try:
            if not Path(self.storage_path).exists():
                return

            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for memory_id, memory_data in data.get('memories', {}).items():
                self.memories[memory_id] = Memory(
                    id=memory_data['id'],
                    content=memory_data['content'],
                    category=memory_data['category'],
                    embedding=memory_data.get('embedding'),
                    created_at=memory_data['created_at'],
                    updated_at=memory_data['updated_at'],
                    access_count=memory_data.get('access_count', 0),
                    metadata=memory_data.get('metadata', {})
                )

            if self.verbose:
                print(f"[MEMORY] Loaded {len(self.memories)} memories")

        except Exception as e:
            if self.verbose:
                print(f"[MEMORY] Load error: {e}")
