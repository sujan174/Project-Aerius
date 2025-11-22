"""
Episodic Memory Store

Provides long-term memory via vector search:
- Store interaction summaries as embeddings
- Retrieve relevant past context
- Semantic search across history

Uses ChromaDB for vector storage and Gemini for embeddings.

Author: AI System
Version: 1.0
"""

import time
import json
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
import google.generativeai as genai


@dataclass
class ConversationSession:
    """A single conversation session summary"""
    session_id: str
    start_time: float
    end_time: float
    summary: str  # Concise summary of what was discussed
    topics: List[str]  # Key topics discussed
    agents_used: List[str]  # Agents that were used
    message_count: int


class ConversationSessionStore:
    """
    Stores concise summaries of the last N conversation sessions.

    Persists to JSON file for cross-session retrieval.
    """

    def __init__(
        self,
        storage_path: str = "data/conversation_sessions.json",
        max_sessions: int = 7,
        verbose: bool = False
    ):
        """
        Initialize conversation session store.

        Args:
            storage_path: Path to JSON file for persistence
            max_sessions: Maximum number of sessions to keep
            verbose: Enable detailed logging
        """
        self.storage_path = storage_path
        self.max_sessions = max_sessions
        self.verbose = verbose

        # Ensure directory exists
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

        # Load existing sessions
        self.sessions: List[Dict] = []
        self._load()

        # Current session data
        self.current_session = {
            'session_id': '',
            'start_time': 0,
            'messages': [],
            'agents_used': set(),
            'topics': set()
        }

        if self.verbose:
            print(f"[SESSIONS] Loaded {len(self.sessions)} previous sessions")

    def start_session(self, session_id: str):
        """Start tracking a new session"""
        self.current_session = {
            'session_id': session_id,
            'start_time': time.time(),
            'messages': [],
            'agents_used': set(),
            'topics': set()
        }

        if self.verbose:
            print(f"[SESSIONS] Started session {session_id[:8]}...")

    def add_message(self, user_message: str, response: str, agents_used: List[str] = None):
        """Add a message exchange to the current session"""
        self.current_session['messages'].append({
            'user': user_message[:500],  # Store more content for better summaries
            'response': response[:500],
            'timestamp': time.time()
        })

        if agents_used:
            self.current_session['agents_used'].update(agents_used)

        # Extract simple topics from user message
        self._extract_topics(user_message)

    def _extract_topics(self, message: str):
        """Extract key topics from a message (simple keyword extraction)"""
        import re

        # Simple topic extraction - look for key nouns/actions
        # Use word boundaries to avoid partial matches
        keywords = {
            'ticket': r'\btickets?\b',
            'issue': r'\bissues?\b',
            'bug': r'\bbugs?\b',
            'feature': r'\bfeatures?\b',
            'meeting': r'\bmeetings?\b',
            'calendar': r'\bcalendar\b',
            'slack': r'\bslack\b',
            'jira': r'\bjira\b',
            'github': r'\bgithub\b',
            'email': r'\bemails?\b',
            'task': r'\btasks?\b',
            'project': r'\bprojects?\b',
            'code': r'\bcode\b',
            'review': r'\breview\b',
            'deploy': r'\bdeploy(?:ment)?\b',
            'test': r'\btests?\b',
            'build': r'\bbuilds?\b',
            'pr': r'\bpull\s*requests?\b|\bPRs?\b',  # Only match "pull request" or "PR"
            'merge': r'\bmerge\b',
        }

        message_lower = message.lower()
        for topic, pattern in keywords.items():
            if re.search(pattern, message_lower, re.IGNORECASE):
                self.current_session['topics'].add(topic)

    async def end_session(self, llm_client=None):
        """End the current session and create a summary"""
        if not self.current_session['messages']:
            return

        # Create summary
        if llm_client and len(self.current_session['messages']) > 0:
            summary = await self._create_session_summary(llm_client)
        else:
            summary = self._create_simple_summary()

        # Create session record with actual message excerpts
        messages = self.current_session['messages']
        user_messages = [msg['user'] for msg in messages[:5]]  # First 5 user messages

        session_record = {
            'session_id': self.current_session['session_id'],
            'start_time': self.current_session['start_time'],
            'end_time': time.time(),
            'summary': summary,
            'topics': list(self.current_session['topics']),
            'agents_used': list(self.current_session['agents_used']),
            'message_count': len(self.current_session['messages']),
            'user_messages': user_messages  # Store actual messages for context
        }

        # Add to sessions list
        self.sessions.append(session_record)

        # Keep only last N sessions
        if len(self.sessions) > self.max_sessions:
            self.sessions = self.sessions[-self.max_sessions:]

        # Save to disk
        self._save()

        if self.verbose:
            print(f"[SESSIONS] Saved session with {session_record['message_count']} messages")

    async def _create_session_summary(self, llm_client) -> str:
        """Create a concise summary using LLM"""
        messages = self.current_session['messages']

        # Build conversation excerpt
        excerpt = []
        for msg in messages[:10]:  # First 10 messages
            excerpt.append(f"User: {msg['user']}")
            excerpt.append(f"Response: {msg['response'][:100]}...")

        prompt = f"""Summarize this conversation session in 2-3 sentences.
Focus on: what tasks were requested, what was accomplished.

Conversation:
{chr(10).join(excerpt)}

Keep the summary under 100 words. Just the summary, no preamble."""

        try:
            response = await llm_client.generate(prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except Exception as e:
            if self.verbose:
                print(f"[SESSIONS] LLM summary failed: {e}")
            return self._create_simple_summary()

    def _create_simple_summary(self) -> str:
        """Create a simple summary without LLM"""
        msg_count = len(self.current_session['messages'])
        topics = list(self.current_session['topics'])[:5]
        agents = list(self.current_session['agents_used'])

        parts = [f"{msg_count} messages"]
        if topics:
            parts.append(f"discussed {', '.join(topics)}")
        if agents:
            parts.append(f"used {', '.join(agents)}")

        return "; ".join(parts)

    def get_recent_sessions(self, n: int = 7) -> List[Dict]:
        """Get the N most recent sessions"""
        return self.sessions[-n:]

    def format_for_prompt(self) -> str:
        """Format recent sessions for system prompt injection"""
        if not self.sessions:
            return ""

        lines = [
            "# Previous Conversation Sessions",
            "",
            "Here are summaries of recent conversations with this user:",
            ""
        ]

        for session in self.sessions[-5:]:  # Last 5 sessions
            date = datetime.fromtimestamp(session['start_time']).strftime("%Y-%m-%d %H:%M")

            lines.append(f"**[{date}]** ({session['message_count']} messages)")
            lines.append(f"   {session['summary']}")

            # Show first user message as context
            user_messages = session.get('user_messages', [])
            if user_messages:
                first_msg = user_messages[0][:80]
                lines.append(f"   _Started with: \"{first_msg}{'...' if len(user_messages[0]) > 80 else ''}\"_")

            if session.get('agents_used'):
                lines.append(f"   _Agents: {', '.join(session['agents_used'])}_")
            lines.append("")

        lines.append("Use this context to understand ongoing work and user preferences.")

        return "\n".join(lines)

    def get_context_for_query(self, query: str) -> str:
        """Get relevant session context for a query like 'what did we discuss yesterday'"""
        if not self.sessions:
            return "No previous conversation sessions found."

        lines = ["Here are your recent conversation sessions:"]

        for session in reversed(self.sessions[-7:]):
            date = datetime.fromtimestamp(session['start_time']).strftime("%B %d, %Y at %I:%M %p")
            duration_mins = int((session['end_time'] - session['start_time']) / 60)

            lines.append(f"\n**{date}** ({duration_mins} min, {session['message_count']} messages)")
            lines.append(f"Summary: {session['summary']}")

            # Show actual user messages if available
            user_messages = session.get('user_messages', [])
            if user_messages:
                lines.append("You asked:")
                for i, msg in enumerate(user_messages[:3], 1):  # Show first 3
                    lines.append(f"  {i}. \"{msg[:100]}{'...' if len(msg) > 100 else ''}\"")

            if session.get('agents_used'):
                lines.append(f"Agents used: {', '.join(session['agents_used'])}")

        return "\n".join(lines)

    def _save(self):
        """Save sessions to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'sessions': self.sessions,
                    'saved_at': time.time()
                }, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[SESSIONS] Save error: {e}")

    def _load(self):
        """Load sessions from disk"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.sessions = data.get('sessions', [])
        except Exception as e:
            if self.verbose:
                print(f"[SESSIONS] Load error: {e}")
            self.sessions = []


@dataclass
class Episode:
    """A single episodic memory entry"""
    id: str  # Unique episode ID
    summary: str  # Natural language summary of interaction
    user_message: str  # Original user message
    agent_response: str  # Summary of agent action/response

    # Metadata for filtering
    timestamp: float
    agents_used: List[str]
    intent_type: str
    entities: List[Dict[str, str]]  # [{type: "ISSUE", value: "KAN-123"}]
    success: bool

    # Computed fields
    embedding: Optional[List[float]] = None


class EpisodicMemoryStore:
    """
    Long-term memory storage using vector embeddings.

    Capabilities:
    - Store interaction summaries
    - Semantic similarity search
    - Metadata-based filtering
    - Automatic summarization
    """

    COLLECTION_NAME = "episodic_memory"
    EMBEDDING_MODEL = "models/text-embedding-004"

    def __init__(
        self,
        persist_directory: str = "data/memory",
        verbose: bool = False
    ):
        """
        Initialize episodic memory store.

        Args:
            persist_directory: Where to store ChromaDB data
            verbose: Enable detailed logging
        """
        self.persist_directory = persist_directory
        self.verbose = verbose

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )

        # Statistics
        self.episodes_stored = 0
        self.queries_made = 0
        self.cache_hits = 0

        # Embedding cache (avoid re-embedding same text)
        self._embedding_cache: Dict[str, List[float]] = {}

        if self.verbose:
            count = self.collection.count()
            print(f"[MEMORY] Initialized with {count} existing episodes")

    async def add_episode(
        self,
        user_message: str,
        agent_response: str,
        agents_used: List[str],
        intent_type: str,
        entities: List[Dict[str, str]],
        success: bool = True,
        llm_client=None  # For summarization
    ) -> str:
        """
        Store a new episodic memory.

        Called at end of each interaction turn.

        Returns:
            Episode ID
        """
        # Generate episode ID
        episode_id = self._generate_id(user_message, time.time())

        # Create summary
        summary = await self._create_summary(
            user_message, agent_response, agents_used, llm_client
        )

        # Create embedding
        embedding = await self._get_embedding(summary)

        # Prepare metadata (ChromaDB requires flat structure)
        metadata = {
            "timestamp": time.time(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "agents": ",".join(agents_used) if agents_used else "",
            "intent": intent_type,
            "entities_json": json.dumps(entities),
            "success": success,
            "user_message_preview": user_message[:200]
        }

        # Store in ChromaDB
        self.collection.add(
            ids=[episode_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[metadata]
        )

        self.episodes_stored += 1

        if self.verbose:
            print(f"[MEMORY] Stored episode {episode_id[:8]}... ({len(summary)} chars)")

        return episode_id

    async def retrieve_relevant(
        self,
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.75,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve relevant past episodes for current query.

        Called at start of each turn to provide context.

        Args:
            query: Current user message
            n_results: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)
            filters: Optional ChromaDB where filters

        Returns:
            List of relevant episodes with similarity scores
        """
        self.queries_made += 1

        # Check if collection is empty
        if self.collection.count() == 0:
            return []

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Build where clause if filters provided
        where = None
        if filters:
            where = filters

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Process results
        episodes = []

        if results and results['ids'] and results['ids'][0]:
            for i, episode_id in enumerate(results['ids'][0]):
                # ChromaDB returns distances, convert to similarity
                # For cosine: similarity = 1 - distance
                distance = results['distances'][0][i]
                similarity = 1 - distance

                if similarity >= min_similarity:
                    episodes.append({
                        'id': episode_id,
                        'summary': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'similarity': round(similarity, 3)
                    })

        if self.verbose:
            print(f"[MEMORY] Retrieved {len(episodes)} relevant episodes (query: {query[:50]}...)")

        return episodes

    def format_for_prompt(self, episodes: List[Dict]) -> str:
        """
        Format retrieved episodes for system prompt injection.

        Args:
            episodes: List of episode dicts from retrieve_relevant()

        Returns:
            Formatted string for prompt
        """
        if not episodes:
            return ""

        lines = [
            "# Relevant Past Interactions",
            "",
            "The following past interactions may be relevant to the current request:",
            ""
        ]

        for i, ep in enumerate(episodes, 1):
            metadata = ep['metadata']
            date = metadata.get('date', 'Unknown')
            agents = metadata.get('agents', 'Unknown')

            lines.append(f"**{i}. [{date}]** (similarity: {ep['similarity']})")
            lines.append(f"   {ep['summary']}")
            lines.append(f"   _Agents: {agents}_")
            lines.append("")

        lines.append("Use this context to inform your response, but don't reference it explicitly unless relevant.")

        return "\n".join(lines)

    async def _create_summary(
        self,
        user_message: str,
        agent_response: str,
        agents_used: List[str],
        llm_client=None
    ) -> str:
        """Create a concise summary of the interaction"""

        if llm_client:
            # Use LLM for better summaries
            prompt = f"""Summarize this interaction in 1-2 sentences for memory retrieval.

User: {user_message}
Response: {agent_response[:500]}
Agents: {', '.join(agents_used) if agents_used else 'system'}

Focus on: what was requested, what was done, and the outcome.
Keep it under 100 words. Just the summary, no preamble."""

            try:
                response = await llm_client.generate(prompt)
                summary = response.text if hasattr(response, 'text') else str(response)
                return summary.strip()
            except Exception as e:
                if self.verbose:
                    print(f"[MEMORY] LLM summary failed: {e}")

        # Fallback: Simple template
        agents_str = ", ".join(agents_used) if agents_used else "system"
        return f"User asked: {user_message[:100]}. {agents_str} responded: {agent_response[:150]}"

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, with caching"""

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            self.cache_hits += 1
            return self._embedding_cache[cache_key]

        # Generate embedding
        try:
            result = genai.embed_content(
                model=self.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']

            # Cache it
            self._embedding_cache[cache_key] = embedding

            # Limit cache size
            if len(self._embedding_cache) > 1000:
                # Remove oldest entries (FIFO)
                keys = list(self._embedding_cache.keys())
                for key in keys[:100]:
                    del self._embedding_cache[key]

            return embedding

        except Exception as e:
            if self.verbose:
                print(f"[MEMORY] Embedding failed: {e}")
            # Return zero vector as fallback (won't match well)
            return [0.0] * 768

    def _generate_id(self, content: str, timestamp: float) -> str:
        """Generate unique episode ID"""
        data = f"{content}{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def get_statistics(self) -> Dict:
        """Get memory store statistics"""
        return {
            'total_episodes': self.collection.count(),
            'episodes_stored_this_session': self.episodes_stored,
            'queries_made': self.queries_made,
            'embedding_cache_size': len(self._embedding_cache),
            'cache_hits': self.cache_hits
        }

    def clear_old_episodes(self, days_to_keep: int = 90):
        """Remove episodes older than specified days"""
        cutoff = time.time() - (days_to_keep * 24 * 60 * 60)

        # Get all IDs with old timestamps
        results = self.collection.get(
            where={"timestamp": {"$lt": cutoff}},
            include=["metadatas"]
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            if self.verbose:
                print(f"[MEMORY] Cleared {len(results['ids'])} old episodes")
