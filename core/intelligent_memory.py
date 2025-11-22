"""
Intelligent Memory System

A smarter memory system that:
- Extracts semantic intents and entities using LLM
- Maintains a knowledge graph of entities (projects, people, tickets)
- Provides intelligent retrieval based on context
- Learns patterns and provides proactive insights

Author: AI System
Version: 2.0
"""

import time
import json
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict


@dataclass
class ExtractedIntent:
    """Structured intent extracted from user message"""
    intent_type: str  # create_ticket, query_status, schedule_meeting, set_preference, etc.
    confidence: float
    entities: List[Dict[str, Any]]  # [{type, value, confidence}]
    requires_action: bool
    sentiment: str  # neutral, frustrated, satisfied, urgent
    context_needed: List[str]  # What context would help


@dataclass
class Entity:
    """An entity in the knowledge graph"""
    id: str
    type: str  # project, person, ticket, channel, etc.
    value: str  # The actual value (KAN-123, @john, #general)
    properties: Dict[str, Any]  # Additional properties
    first_seen: float
    last_seen: float
    mention_count: int
    relationships: List[Dict[str, str]]  # [{relation, target_id}]


class SemanticExtractor:
    """
    Extracts semantic meaning from messages using LLM.

    This is the key improvement - instead of regex keywords,
    we understand what the user actually means.
    """

    def __init__(self, llm_client, verbose: bool = False):
        self.llm = llm_client
        self.verbose = verbose
        self._cache: Dict[str, ExtractedIntent] = {}

    async def extract(self, message: str) -> ExtractedIntent:
        """Extract structured intent and entities from message"""

        # Check cache
        cache_key = hashlib.md5(message.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = f"""Analyze this user message and extract structured information.

Message: "{message}"

Return ONLY valid JSON (no markdown, no explanation):
{{
    "intent_type": "create_ticket|query_status|schedule_meeting|send_message|set_preference|ask_question|greeting|other",
    "confidence": 0.0-1.0,
    "entities": [
        {{"type": "project|person|ticket|channel|date|time|tool", "value": "extracted value", "confidence": 0.9}}
    ],
    "requires_action": true/false,
    "sentiment": "neutral|frustrated|satisfied|urgent",
    "context_needed": ["previous_ticket", "user_preferences", "project_defaults"]
}}

Examples:
- "create a jira ticket for login bug in KAN" -> intent: create_ticket, entities: [project:KAN, issue_type:bug, summary:login bug]
- "what time is it" -> intent: ask_question, entities: [topic:time], requires_action: false
- "my name is John" -> intent: set_preference, entities: [preference:user_name, value:John]
- "send a slack message to #general" -> intent: send_message, entities: [channel:#general, tool:slack]"""

        try:
            response = await self.llm.generate(prompt)
            text = response.text if hasattr(response, 'text') else str(response)

            # Parse JSON
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])

                intent = ExtractedIntent(
                    intent_type=data.get('intent_type', 'other'),
                    confidence=data.get('confidence', 0.5),
                    entities=data.get('entities', []),
                    requires_action=data.get('requires_action', False),
                    sentiment=data.get('sentiment', 'neutral'),
                    context_needed=data.get('context_needed', [])
                )

                # Cache it
                self._cache[cache_key] = intent

                if self.verbose:
                    print(f"[SEMANTIC] Extracted: {intent.intent_type} with {len(intent.entities)} entities")

                return intent

        except Exception as e:
            if self.verbose:
                print(f"[SEMANTIC] Extraction failed: {e}")

        # Fallback
        return ExtractedIntent(
            intent_type='other',
            confidence=0.3,
            entities=[],
            requires_action=False,
            sentiment='neutral',
            context_needed=[]
        )


class EntityStore:
    """
    Knowledge graph of entities the user has interacted with.

    Stores: projects, people, tickets, channels, etc.
    Tracks relationships and usage patterns.
    """

    def __init__(self, storage_path: str = "data/entity_store.json", verbose: bool = False):
        self.storage_path = storage_path
        self.verbose = verbose
        self.entities: Dict[str, Entity] = {}
        self._load()

    def add_or_update(self, entity_type: str, value: str, properties: Dict = None) -> Entity:
        """Add or update an entity"""
        entity_id = f"{entity_type}:{value.lower()}"

        if entity_id in self.entities:
            # Update existing
            entity = self.entities[entity_id]
            entity.last_seen = time.time()
            entity.mention_count += 1
            if properties:
                entity.properties.update(properties)
        else:
            # Create new
            entity = Entity(
                id=entity_id,
                type=entity_type,
                value=value,
                properties=properties or {},
                first_seen=time.time(),
                last_seen=time.time(),
                mention_count=1,
                relationships=[]
            )
            self.entities[entity_id] = entity

            if self.verbose:
                print(f"[ENTITIES] New entity: {entity_id}")

        self._save()
        return entity

    def add_relationship(self, from_id: str, relation: str, to_id: str):
        """Add a relationship between entities"""
        if from_id in self.entities:
            self.entities[from_id].relationships.append({
                'relation': relation,
                'target': to_id
            })
            self._save()

    def get(self, entity_type: str, value: str) -> Optional[Entity]:
        """Get an entity by type and value"""
        entity_id = f"{entity_type}:{value.lower()}"
        return self.entities.get(entity_id)

    def get_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a type"""
        return [e for e in self.entities.values() if e.type == entity_type]

    def get_recent(self, limit: int = 10) -> List[Entity]:
        """Get recently used entities"""
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: e.last_seen,
            reverse=True
        )
        return sorted_entities[:limit]

    def get_frequent(self, entity_type: str = None, limit: int = 5) -> List[Entity]:
        """Get frequently used entities"""
        entities = self.entities.values()
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]

        sorted_entities = sorted(
            entities,
            key=lambda e: e.mention_count,
            reverse=True
        )
        return sorted_entities[:limit]

    def format_for_prompt(self) -> str:
        """Format entity knowledge for system prompt"""
        if not self.entities:
            return ""

        lines = [
            "# User's Known Entities",
            "",
            "Based on past interactions, here are entities the user works with:",
            ""
        ]

        # Group by type
        by_type = defaultdict(list)
        for entity in self.entities.values():
            by_type[entity.type].append(entity)

        for entity_type, entities in by_type.items():
            # Sort by frequency
            sorted_entities = sorted(entities, key=lambda e: e.mention_count, reverse=True)[:5]

            lines.append(f"**{entity_type.title()}s:**")
            for entity in sorted_entities:
                props = ""
                if entity.properties:
                    props = f" ({', '.join(f'{k}={v}' for k, v in list(entity.properties.items())[:2])})"
                lines.append(f"  - {entity.value}{props} [used {entity.mention_count}x]")
            lines.append("")

        return "\n".join(lines)

    def _save(self):
        """Save to disk"""
        try:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert entities to serializable format
            data = {}
            for entity_id, entity in self.entities.items():
                data[entity_id] = {
                    'id': entity.id,
                    'type': entity.type,
                    'value': entity.value,
                    'properties': entity.properties,
                    'first_seen': entity.first_seen,
                    'last_seen': entity.last_seen,
                    'mention_count': entity.mention_count,
                    'relationships': entity.relationships
                }

            with open(self.storage_path, 'w') as f:
                json.dump({'entities': data, 'saved_at': time.time()}, f, indent=2)

        except Exception as e:
            if self.verbose:
                print(f"[ENTITIES] Save error: {e}")

    def _load(self):
        """Load from disk"""
        try:
            if Path(self.storage_path).exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)

                for entity_id, entity_data in data.get('entities', {}).items():
                    self.entities[entity_id] = Entity(
                        id=entity_data['id'],
                        type=entity_data['type'],
                        value=entity_data['value'],
                        properties=entity_data.get('properties', {}),
                        first_seen=entity_data.get('first_seen', time.time()),
                        last_seen=entity_data.get('last_seen', time.time()),
                        mention_count=entity_data.get('mention_count', 1),
                        relationships=entity_data.get('relationships', [])
                    )

                if self.verbose:
                    print(f"[ENTITIES] Loaded {len(self.entities)} entities")

        except Exception as e:
            if self.verbose:
                print(f"[ENTITIES] Load error: {e}")


class IntelligentSession:
    """
    A smarter session that tracks structured data, not just messages.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        self.interactions: List[Dict] = []
        self.entities_mentioned: Dict[str, int] = {}  # entity_id -> count
        self.intents: List[str] = []
        self.actions_taken: List[Dict] = []
        self.sentiment_history: List[str] = []

    def add_interaction(
        self,
        user_message: str,
        response: str,
        intent: ExtractedIntent,
        agents_used: List[str] = None,
        action_result: Dict = None
    ):
        """Add an interaction with structured data"""
        self.interactions.append({
            'user_message': user_message,
            'response': response[:500],
            'intent': intent.intent_type,
            'entities': intent.entities,
            'agents': agents_used or [],
            'timestamp': time.time()
        })

        # Track entities
        for entity in intent.entities:
            entity_id = f"{entity['type']}:{entity['value'].lower()}"
            self.entities_mentioned[entity_id] = self.entities_mentioned.get(entity_id, 0) + 1

        # Track intents
        if intent.intent_type not in self.intents:
            self.intents.append(intent.intent_type)

        # Track sentiment
        self.sentiment_history.append(intent.sentiment)

        # Track actions
        if agents_used:
            self.actions_taken.append({
                'agents': agents_used,
                'intent': intent.intent_type,
                'result': action_result,
                'timestamp': time.time()
            })

    def get_summary_data(self) -> Dict:
        """Get structured summary data for this session"""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'end_time': time.time(),
            'interaction_count': len(self.interactions),
            'intents': self.intents,
            'top_entities': sorted(
                self.entities_mentioned.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'agents_used': list(set(
                agent for interaction in self.interactions
                for agent in interaction.get('agents', [])
            )),
            'sentiment_trend': self._analyze_sentiment_trend(),
            'user_messages': [i['user_message'][:200] for i in self.interactions[:5]]
        }

    def _analyze_sentiment_trend(self) -> str:
        """Analyze overall sentiment trend"""
        if not self.sentiment_history:
            return 'neutral'

        # Simple analysis - check if sentiment improved or degraded
        sentiments = {'frustrated': -1, 'neutral': 0, 'satisfied': 1, 'urgent': 0}
        scores = [sentiments.get(s, 0) for s in self.sentiment_history]

        if len(scores) < 2:
            return self.sentiment_history[-1]

        first_half = sum(scores[:len(scores)//2]) / max(1, len(scores)//2)
        second_half = sum(scores[len(scores)//2:]) / max(1, len(scores) - len(scores)//2)

        if second_half > first_half + 0.3:
            return 'improving'
        elif second_half < first_half - 0.3:
            return 'degrading'
        else:
            return self.sentiment_history[-1]


class IntelligentMemory:
    """
    Main intelligent memory system.

    Combines:
    - Semantic extraction
    - Entity store (knowledge graph)
    - Session tracking
    - Pattern detection
    """

    def __init__(
        self,
        llm_client,
        storage_dir: str = "data/intelligent_memory",
        verbose: bool = False
    ):
        self.llm = llm_client
        self.storage_dir = storage_dir
        self.verbose = verbose

        # Ensure directory exists
        Path(storage_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.extractor = SemanticExtractor(llm_client, verbose)
        self.entity_store = EntityStore(
            storage_path=f"{storage_dir}/entities.json",
            verbose=verbose
        )

        # Session storage
        self.sessions_path = f"{storage_dir}/sessions.json"
        self.past_sessions: List[Dict] = []
        self._load_sessions()

        # Current session
        self.current_session: Optional[IntelligentSession] = None

        # Pattern detection
        self.patterns: Dict[str, List[Dict]] = defaultdict(list)

        if self.verbose:
            print(f"[MEMORY] Initialized with {len(self.entity_store.entities)} entities, {len(self.past_sessions)} sessions")

    def start_session(self, session_id: str):
        """Start a new session"""
        self.current_session = IntelligentSession(session_id)

        if self.verbose:
            print(f"[MEMORY] Started session {session_id[:8]}...")

    async def process_message(
        self,
        user_message: str,
        response: str,
        agents_used: List[str] = None,
        action_result: Dict = None
    ) -> ExtractedIntent:
        """
        Process a message exchange.

        Returns the extracted intent for use by the orchestrator.
        """
        # Extract semantic intent
        intent = await self.extractor.extract(user_message)

        # Update entity store
        for entity_data in intent.entities:
            entity = self.entity_store.add_or_update(
                entity_type=entity_data['type'],
                value=entity_data['value'],
                properties={'confidence': entity_data.get('confidence', 0.5)}
            )

            # Track relationships (e.g., ticket belongs to project)
            if entity_data['type'] == 'ticket' and any(e['type'] == 'project' for e in intent.entities):
                project = next(e for e in intent.entities if e['type'] == 'project')
                self.entity_store.add_relationship(
                    entity.id,
                    'belongs_to',
                    f"project:{project['value'].lower()}"
                )

        # Add to current session
        if self.current_session:
            self.current_session.add_interaction(
                user_message=user_message,
                response=response,
                intent=intent,
                agents_used=agents_used,
                action_result=action_result
            )

        # Detect patterns
        self._detect_patterns(intent, agents_used)

        return intent

    async def end_session(self):
        """End the current session and create summary"""
        if not self.current_session or not self.current_session.interactions:
            return

        # Get structured summary data
        summary_data = self.current_session.get_summary_data()

        # Create natural language summary
        summary_data['summary'] = await self._create_session_summary(summary_data)

        # Add to past sessions
        self.past_sessions.append(summary_data)

        # Keep only last 10 sessions
        if len(self.past_sessions) > 10:
            self.past_sessions = self.past_sessions[-10:]

        # Save
        self._save_sessions()

        if self.verbose:
            print(f"[MEMORY] Ended session with {summary_data['interaction_count']} interactions")

        self.current_session = None

    async def _create_session_summary(self, summary_data: Dict) -> str:
        """Create natural language summary from structured data"""

        # Build a factual prompt
        entities_str = ", ".join([f"{e[0]}" for e in summary_data.get('top_entities', [])])
        intents_str = ", ".join(summary_data.get('intents', []))
        agents_str = ", ".join(summary_data.get('agents_used', []))
        messages = summary_data.get('user_messages', [])

        prompt = f"""Create a 2-3 sentence summary of this conversation session.

Facts:
- User performed these actions: {intents_str}
- Entities discussed: {entities_str}
- Agents used: {agents_str or 'none'}
- Number of exchanges: {summary_data['interaction_count']}
- User messages: {messages[:3]}

Write a factual summary. Do not invent details. Just the summary, no preamble."""

        try:
            response = await self.llm.generate(prompt)
            return response.text.strip() if hasattr(response, 'text') else str(response).strip()
        except:
            # Fallback
            return f"{summary_data['interaction_count']} interactions involving {intents_str}"

    def _detect_patterns(self, intent: ExtractedIntent, agents_used: List[str]):
        """Detect usage patterns for proactive insights"""

        # Track intent patterns by time
        hour = datetime.now().hour
        day = datetime.now().weekday()

        pattern_key = f"{intent.intent_type}_{day}_{hour}"
        self.patterns[pattern_key].append({
            'timestamp': time.time(),
            'entities': intent.entities,
            'agents': agents_used
        })

        # Track agent preferences by intent
        if agents_used:
            for agent in agents_used:
                preference_key = f"agent_for_{intent.intent_type}"
                self.patterns[preference_key].append({
                    'timestamp': time.time(),
                    'agent': agent
                })

    def get_context_for_message(self, message: str) -> str:
        """Get relevant context for a new message"""

        lines = []

        # Entity context
        entity_context = self.entity_store.format_for_prompt()
        if entity_context:
            lines.append(entity_context)

        # Recent session context
        if self.past_sessions:
            lines.append("# Recent Sessions")
            lines.append("")

            for session in self.past_sessions[-3:]:
                date = datetime.fromtimestamp(session['start_time']).strftime("%Y-%m-%d %H:%M")
                lines.append(f"**[{date}]** ({session['interaction_count']} interactions)")
                lines.append(f"   {session.get('summary', 'No summary')}")

                # Show what user actually asked
                if session.get('user_messages'):
                    first_msg = session['user_messages'][0][:80]
                    lines.append(f"   _Started with: \"{first_msg}...\"_")

                lines.append("")

        return "\n".join(lines)

    def get_proactive_insights(self) -> List[Dict]:
        """Get proactive insights based on patterns"""
        insights = []

        # Check for repeated actions
        hour = datetime.now().hour
        day = datetime.now().weekday()

        for pattern_key, occurrences in self.patterns.items():
            if len(occurrences) >= 3:
                # Check if this is a time-based pattern
                if f"_{day}_" in pattern_key:
                    intent_type = pattern_key.split('_')[0]
                    insights.append({
                        'type': 'time_pattern',
                        'message': f"You often {intent_type} at this time. Need help with that?",
                        'confidence': min(0.9, len(occurrences) * 0.2)
                    })

        return insights

    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_entities': len(self.entity_store.entities),
            'total_sessions': len(self.past_sessions),
            'patterns_detected': len(self.patterns),
            'current_session_interactions': len(self.current_session.interactions) if self.current_session else 0
        }

    def _save_sessions(self):
        """Save sessions to disk"""
        try:
            with open(self.sessions_path, 'w') as f:
                json.dump({
                    'sessions': self.past_sessions,
                    'saved_at': time.time()
                }, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[MEMORY] Save error: {e}")

    def _load_sessions(self):
        """Load sessions from disk"""
        try:
            if Path(self.sessions_path).exists():
                with open(self.sessions_path, 'r') as f:
                    data = json.load(f)
                    self.past_sessions = data.get('sessions', [])
        except Exception as e:
            if self.verbose:
                print(f"[MEMORY] Load error: {e}")
