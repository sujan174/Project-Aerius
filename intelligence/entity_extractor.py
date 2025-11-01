"""
Entity Extraction System

Extracts structured information from natural language:
- Projects (KAN, PROJ-*, repo names)
- People (@mentions, names, teams)
- Dates (tomorrow, next sprint, Friday)
- Priorities (critical, high, low)
- Resources (issues, PRs, URLs, IDs)

Author: AI System
Version: 2.0
"""

import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from .base_types import Entity, EntityType


class EntityExtractor:
    """
    Extract entities from natural language

    Recognizes:
    - Projects: KAN, PROJ-*, repository names
    - People: @username, @team, names
    - Dates: tomorrow, next week, by Friday, 2024-01-15
    - Priorities: critical, high, medium, low
    - Resources: KAN-123, #456, PR #789
    - Teams: @engineering, security team
    - Channels: #general, #bugs
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Regex patterns for entity extraction
        self.patterns = {
            # Jira-style issues: KAN-123, PROJ-456
            EntityType.ISSUE: [
                r'\b([A-Z]{2,10}-\d+)\b',
                r'\bissue\s+(\d+)\b',
                r'\bticket\s+(\d+)\b',
            ],

            # GitHub PR: PR #123, #456
            EntityType.PR: [
                r'\bPR\s*#?(\d+)\b',
                r'\bpull request\s*#?(\d+)\b',
                r'#(\d+)',
            ],

            # Projects: Uppercase words or specific patterns
            EntityType.PROJECT: [
                r'\b([A-Z]{2,10})\s+project\b',
                r'\bproject\s+([A-Z]{2,10})\b',
            ],

            # People: @username, @firstname.lastname
            EntityType.PERSON: [
                r'@([\w.-]+)',
                r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            ],

            # Teams: @team-name, security team
            EntityType.TEAM: [
                r'@([\w-]+team)\b',
                r'\b([\w-]+\s+team)\b',
                r'@(engineering|security|devops|qa|design)\b',
            ],

            # Channels: #channel-name
            EntityType.CHANNEL: [
                r'#([\w-]+)',
                r'\bchannel\s+([\w-]+)\b',
            ],

            # Repositories: owner/repo
            EntityType.REPOSITORY: [
                r'\b([\w-]+)/([\w-]+)\b',
                r'\brepo\s+([\w-]+)\b',
                r'\brepository\s+([\w-]+)\b',
            ],

            # Files: path/to/file.ext
            EntityType.FILE: [
                r'([\w/-]+\.[\w]+)',
                r'\bfile\s+([\w/.]+)\b',
            ],

            # URLs
            EntityType.RESOURCE: [
                r'(https?://[^\s]+)',
            ],
        }

        # Priority keywords
        self.priority_keywords = {
            'critical': ['critical', 'blocker', 'urgent', 'emergency'],
            'high': ['high', 'important', 'priority', 'soon'],
            'medium': ['medium', 'normal', 'standard'],
            'low': ['low', 'minor', 'trivial', 'nice to have', 'nice-to-have']
        }

        # Status keywords
        self.status_keywords = {
            'open': ['open', 'new', 'todo', 'backlog'],
            'in_progress': ['in progress', 'in-progress', 'working', 'started', 'active'],
            'review': ['review', 'reviewing', 'pending review'],
            'done': ['done', 'completed', 'closed', 'resolved', 'fixed'],
            'blocked': ['blocked', 'waiting', 'on hold']
        }

        # Date patterns
        self.date_patterns = {
            'tomorrow': lambda: datetime.now() + timedelta(days=1),
            'today': lambda: datetime.now(),
            'yesterday': lambda: datetime.now() - timedelta(days=1),
            'next week': lambda: datetime.now() + timedelta(weeks=1),
            'next month': lambda: datetime.now() + timedelta(days=30),
        }

        # Weekdays
        self.weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }

    def extract(self, message: str, context: Optional[Dict] = None) -> List[Entity]:
        """
        Extract all entities from message

        Args:
            message: User message to extract from
            context: Optional context (current project, user, etc.)

        Returns:
            List of extracted entities
        """
        entities = []

        # Extract pattern-based entities
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    value = match.group(1) if match.groups() else match.group(0)

                    # Filter out false positives
                    if self._is_valid_entity(value, entity_type):
                        entity = Entity(
                            type=entity_type,
                            value=value,
                            confidence=0.9,
                            context=match.group(0)
                        )
                        entities.append(entity)

        # Extract keyword-based entities
        entities.extend(self._extract_priorities(message))
        entities.extend(self._extract_statuses(message))
        entities.extend(self._extract_dates(message))
        entities.extend(self._extract_labels(message))

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        # Normalize entity values
        for entity in entities:
            entity.normalized_value = self._normalize_value(entity)

        if self.verbose:
            print(f"[ENTITY] Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity}")

        return entities

    def extract_by_type(self, message: str, entity_type: EntityType) -> List[Entity]:
        """Extract only entities of specific type"""
        all_entities = self.extract(message)
        return [e for e in all_entities if e.type == entity_type]

    def find_entity_value(self, entities: List[Entity], entity_type: EntityType) -> Optional[str]:
        """Find first entity value of specific type"""
        for entity in entities:
            if entity.type == entity_type:
                return entity.normalized_value or entity.value
        return None

    def _is_valid_entity(self, value: str, entity_type: EntityType) -> bool:
        """Validate if extracted value is actually an entity"""
        # Filter out too short values
        if len(value) < 2:
            return False

        # Filter out common false positives
        false_positives = {'the', 'it', 'and', 'or', 'in', 'on', 'at'}
        if value.lower() in false_positives:
            return False

        # Type-specific validation
        if entity_type == EntityType.PROJECT:
            # Must be uppercase and reasonable length
            return value.isupper() and 2 <= len(value) <= 10

        elif entity_type == EntityType.ISSUE:
            # Must contain at least one digit
            return any(c.isdigit() for c in value)

        return True

    def _extract_priorities(self, message: str) -> List[Entity]:
        """Extract priority entities"""
        message_lower = message.lower()
        entities = []

        for priority, keywords in self.priority_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    entities.append(Entity(
                        type=EntityType.PRIORITY,
                        value=priority,
                        confidence=0.95,
                        context=keyword
                    ))
                    break  # One priority per level

        return entities

    def _extract_statuses(self, message: str) -> List[Entity]:
        """Extract status entities"""
        message_lower = message.lower()
        entities = []

        for status, keywords in self.status_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    entities.append(Entity(
                        type=EntityType.STATUS,
                        value=status,
                        confidence=0.90,
                        context=keyword
                    ))
                    break

        return entities

    def _extract_dates(self, message: str) -> List[Entity]:
        """Extract date entities"""
        message_lower = message.lower()
        entities = []

        # Check relative dates (tomorrow, next week, etc.)
        for date_phrase, date_func in self.date_patterns.items():
            if date_phrase in message_lower:
                date_value = date_func()
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=date_phrase,
                    confidence=0.95,
                    normalized_value=date_value.strftime('%Y-%m-%d'),
                    context=date_phrase
                ))

        # Check weekdays (next Friday, by Monday, etc.)
        for weekday, weekday_num in self.weekdays.items():
            pattern = r'\b(?:next|by|on)\s+' + weekday + r'\b'
            if re.search(pattern, message_lower):
                # Calculate next occurrence of this weekday
                today = datetime.now()
                days_ahead = weekday_num - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                next_date = today + timedelta(days=days_ahead)

                entities.append(Entity(
                    type=EntityType.DATE,
                    value=weekday,
                    confidence=0.90,
                    normalized_value=next_date.strftime('%Y-%m-%d'),
                    context=f"next {weekday}"
                ))

        # Check absolute dates (2024-01-15, 01/15/2024)
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{2}/\d{2}/\d{4})\b',  # MM/DD/YYYY
        ]
        for pattern in date_patterns:
            matches = re.finditer(pattern, message)
            for match in matches:
                entities.append(Entity(
                    type=EntityType.DATE,
                    value=match.group(1),
                    confidence=1.0,
                    normalized_value=match.group(1),
                    context=match.group(0)
                ))

        return entities

    def _extract_labels(self, message: str) -> List[Entity]:
        """Extract label/tag entities"""
        # Labels are often prefixed with # (but not channel names)
        pattern = r'#([a-z][\w-]{2,})'  # lowercase start = likely a label
        matches = re.finditer(pattern, message, re.IGNORECASE)

        entities = []
        for match in matches:
            # Filter out likely channel names (already extracted)
            value = match.group(1)
            if not value.startswith(('general', 'random', 'eng', 'dev')):
                entities.append(Entity(
                    type=EntityType.LABEL,
                    value=value,
                    confidence=0.70,
                    context=match.group(0)
                ))

        return entities

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        seen = set()
        unique = []

        for entity in entities:
            # Create unique key
            key = (entity.type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique.append(entity)

        return unique

    def _normalize_value(self, entity: Entity) -> str:
        """Normalize entity value for consistency"""
        if entity.type == EntityType.PRIORITY:
            # Standardize priority names
            return entity.value.lower()

        elif entity.type == EntityType.STATUS:
            # Standardize status names
            return entity.value.replace(' ', '_').lower()

        elif entity.type == EntityType.PERSON:
            # Remove @ prefix
            return entity.value.lstrip('@')

        elif entity.type == EntityType.CHANNEL:
            # Remove # prefix
            return entity.value.lstrip('#')

        elif entity.type == EntityType.TEAM:
            # Standardize team names
            return entity.value.replace(' ', '-').lower().lstrip('@')

        else:
            return entity.value

    def group_by_type(self, entities: List[Entity]) -> Dict[EntityType, List[Entity]]:
        """Group entities by type"""
        grouped = {}
        for entity in entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity)
        return grouped

    def has_entity_type(self, entities: List[Entity], entity_type: EntityType) -> bool:
        """Check if specific entity type is present"""
        return any(e.type == entity_type for e in entities)

    def get_entity_summary(self, entities: List[Entity]) -> str:
        """Get human-readable summary of extracted entities"""
        if not entities:
            return "No entities extracted"

        grouped = self.group_by_type(entities)
        summary_parts = []

        for entity_type, ents in grouped.items():
            values = [e.value for e in ents]
            summary_parts.append(f"{entity_type.value}: {', '.join(values)}")

        return "; ".join(summary_parts)
