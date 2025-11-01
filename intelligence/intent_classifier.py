"""
Intent Classification Engine

Understands what users really want from their natural language requests.
Classifies intents, detects implicit requirements, and handles multi-intent scenarios.

Author: AI System
Version: 2.0
"""

import re
from typing import List, Dict, Set, Optional
from .base_types import Intent, IntentType


class IntentClassifier:
    """
    Classify user intents from natural language

    Understands:
    - Primary intent (CREATE, READ, UPDATE, DELETE, ANALYZE, COORDINATE)
    - Multiple intents in one request
    - Implicit requirements
    - Contextual indicators
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Intent keyword mappings
        self.intent_keywords = {
            IntentType.CREATE: {
                'primary': ['create', 'make', 'add', 'new', 'start', 'open', 'initialize', 'build', 'generate'],
                'secondary': ['set up', 'spin up', 'kick off'],
                'modifiers': ['issue', 'ticket', 'pr', 'page', 'task', 'project']
            },
            IntentType.READ: {
                'primary': ['show', 'get', 'find', 'search', 'list', 'what', 'where', 'display', 'fetch', 'retrieve'],
                'secondary': ['look up', 'check out', 'pull up'],
                'modifiers': ['status', 'details', 'info', 'information']
            },
            IntentType.UPDATE: {
                'primary': ['update', 'change', 'modify', 'edit', 'fix', 'correct', 'adjust', 'set'],
                'secondary': ['move to', 'transition', 'reassign'],
                'modifiers': ['priority', 'status', 'assignee', 'description']
            },
            IntentType.DELETE: {
                'primary': ['delete', 'remove', 'close', 'archive', 'cancel', 'drop'],
                'secondary': ['get rid of', 'clean up'],
                'modifiers': []
            },
            IntentType.ANALYZE: {
                'primary': ['analyze', 'review', 'check', 'inspect', 'examine', 'evaluate', 'assess', 'audit'],
                'secondary': ['look at', 'take a look'],
                'modifiers': ['code', 'security', 'performance', 'quality']
            },
            IntentType.COORDINATE: {
                'primary': ['notify', 'tell', 'inform', 'alert', 'message', 'ping', 'send', 'post'],
                'secondary': ['let know', 'reach out'],
                'modifiers': ['team', 'channel', 'person', 'slack', 'email']
            },
            IntentType.SEARCH: {
                'primary': ['search', 'find', 'lookup', 'query', 'locate'],
                'secondary': ['look for', 'hunt for'],
                'modifiers': ['for', 'about', 'related to']
            },
            IntentType.WORKFLOW: {
                'primary': ['when', 'if', 'then', 'automate', 'trigger', 'schedule'],
                'secondary': ['set up automation', 'create workflow'],
                'modifiers': []
            }
        }

        # Implicit requirement patterns
        self.implicit_patterns = {
            'urgency': {
                'critical': ['urgent', 'critical', 'asap', 'immediately', 'emergency', 'blocker'],
                'high': ['important', 'soon', 'quickly', 'high priority'],
                'normal': []
            },
            'scope': {
                'single': ['the', 'this', 'that', 'one'],
                'multiple': ['all', 'every', 'each', 'multiple'],
                'batch': ['batch', 'bulk', 'mass']
            },
            'visibility': {
                'public': ['public', 'everyone', 'all', 'team'],
                'private': ['private', 'just me', 'personal']
            }
        }

    def classify(self, message: str) -> List[Intent]:
        """
        Classify intents in user message

        Args:
            message: User message to classify

        Returns:
            List of detected intents with confidence scores
        """
        message_lower = message.lower()
        detected_intents = []

        # Check each intent type
        for intent_type, keywords in self.intent_keywords.items():
            confidence = self._calculate_intent_confidence(message_lower, keywords)

            if confidence > 0.3:  # Threshold for detection
                intent = Intent(
                    type=intent_type,
                    confidence=confidence,
                    raw_indicators=self._extract_indicators(message_lower, keywords)
                )
                detected_intents.append(intent)

        # Sort by confidence
        detected_intents.sort(key=lambda x: x.confidence, reverse=True)

        # Detect implicit requirements
        implicit_reqs = self._detect_implicit_requirements(message_lower)
        for intent in detected_intents:
            intent.implicit_requirements = implicit_reqs

        # If no intents detected, mark as unknown
        if not detected_intents:
            detected_intents.append(Intent(
                type=IntentType.UNKNOWN,
                confidence=0.5,
                implicit_requirements=implicit_reqs
            ))

        if self.verbose:
            print(f"[INTENT] Detected {len(detected_intents)} intents:")
            for intent in detected_intents:
                print(f"  - {intent}")

        return detected_intents

    def get_primary_intent(self, intents: List[Intent]) -> Intent:
        """Get the primary (highest confidence) intent"""
        return intents[0] if intents else Intent(type=IntentType.UNKNOWN, confidence=0.0)

    def has_intent_type(self, intents: List[Intent], intent_type: IntentType) -> bool:
        """Check if specific intent type is present"""
        return any(i.type == intent_type for i in intents)

    def _calculate_intent_confidence(self, message: str, keywords: Dict[str, List[str]]) -> float:
        """
        Calculate confidence score for an intent

        Factors:
        - Primary keyword match (highest weight)
        - Secondary keyword match (medium weight)
        - Modifier presence (context boost)
        - Position in sentence (earlier = higher confidence)
        """
        score = 0.0

        # Check primary keywords (weight: 1.0)
        for keyword in keywords.get('primary', []):
            if keyword in message:
                # Bonus for early position
                position = message.find(keyword)
                position_factor = 1.0 - (position / len(message)) * 0.2

                score = max(score, 0.9 * position_factor)
                break

        # Check secondary keywords (weight: 0.7)
        for keyword in keywords.get('secondary', []):
            if keyword in message:
                score = max(score, 0.7)
                break

        # Boost if modifiers present (context)
        modifiers_found = sum(1 for mod in keywords.get('modifiers', []) if mod in message)
        if modifiers_found > 0:
            score = min(score + (modifiers_found * 0.1), 1.0)

        return score

    def _extract_indicators(self, message: str, keywords: Dict[str, List[str]]) -> List[str]:
        """Extract words that indicated this intent"""
        indicators = []

        for keyword in keywords.get('primary', []) + keywords.get('secondary', []):
            if keyword in message:
                indicators.append(keyword)

        return indicators

    def _detect_implicit_requirements(self, message: str) -> List[str]:
        """
        Detect implicit requirements from message

        Examples:
        - "urgent bug" → implicit: high priority
        - "notify everyone" → implicit: public visibility
        - "create all the issues" → implicit: batch operation
        """
        requirements = []

        # Urgency detection
        for level, keywords in self.implicit_patterns['urgency'].items():
            for keyword in keywords:
                if keyword in message:
                    if level == 'critical':
                        requirements.append('priority:critical')
                        requirements.append('urgent:true')
                    elif level == 'high':
                        requirements.append('priority:high')
                    break

        # Scope detection
        for scope, keywords in self.implicit_patterns['scope'].items():
            for keyword in keywords:
                if keyword in message:
                    requirements.append(f'scope:{scope}')
                    if scope in ['multiple', 'batch']:
                        requirements.append('batch_operation:true')
                    break

        # Visibility detection
        for visibility, keywords in self.implicit_patterns['visibility'].items():
            for keyword in keywords:
                if keyword in message:
                    requirements.append(f'visibility:{visibility}')
                    break

        # Security-related detection
        security_keywords = ['security', 'secure', 'auth', 'authentication', 'authorization', 'permission']
        if any(kw in message for kw in security_keywords):
            requirements.append('security_sensitive:true')

        # Performance-related detection
        performance_keywords = ['performance', 'slow', 'fast', 'optimize', 'speed']
        if any(kw in message for kw in performance_keywords):
            requirements.append('performance_related:true')

        return requirements

    def is_multi_intent(self, intents: List[Intent]) -> bool:
        """Check if message contains multiple high-confidence intents"""
        high_confidence_intents = [i for i in intents if i.confidence > 0.6]
        return len(high_confidence_intents) > 1

    def suggest_clarifications(self, intents: List[Intent]) -> List[str]:
        """
        Suggest what clarifications might be needed based on intents

        Returns:
            List of clarification questions
        """
        clarifications = []

        primary = self.get_primary_intent(intents)

        # CREATE intent clarifications
        if primary.type == IntentType.CREATE:
            clarifications.extend([
                "What should be created? (issue, PR, page, etc.)",
                "Which project/repository?",
                "Any specific details or description?"
            ])

        # READ intent clarifications
        elif primary.type == IntentType.READ:
            clarifications.extend([
                "What information are you looking for?",
                "Which project/repository?"
            ])

        # UPDATE intent clarifications
        elif primary.type == IntentType.UPDATE:
            clarifications.extend([
                "Which resource to update?",
                "What changes should be made?"
            ])

        # COORDINATE intent clarifications
        elif primary.type == IntentType.COORDINATE:
            clarifications.extend([
                "Who should be notified?",
                "What message to send?"
            ])

        return clarifications

    def extract_action_target(self, message: str) -> Optional[str]:
        """
        Extract the target of an action from message

        Examples:
        - "create an issue" → "issue"
        - "update the PR" → "PR"
        - "review the code" → "code"
        """
        message_lower = message.lower()

        # Common targets
        targets = [
            'issue', 'ticket', 'pr', 'pull request', 'page', 'task',
            'project', 'repository', 'repo', 'file', 'code',
            'message', 'channel', 'comment', 'branch'
        ]

        for target in targets:
            if target in message_lower:
                return target

        return None

    def detect_conditional_logic(self, message: str) -> bool:
        """
        Detect if message contains conditional/workflow logic

        Examples:
        - "when X happens, do Y"
        - "if status is done, then notify"
        """
        message_lower = message.lower()

        conditional_patterns = [
            r'\bwhen\b.*\b(then|do|notify|create)',
            r'\bif\b.*\b(then|do|notify|create)',
            r'\bwhenever\b.*\b(then|do|notify|create)',
        ]

        return any(re.search(pattern, message_lower) for pattern in conditional_patterns)
