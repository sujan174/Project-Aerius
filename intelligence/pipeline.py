# Intelligence Pipeline - Task Decomposition and Confidence Scoring

import re
import math
from typing import List, Dict, Set, Optional, Any, Tuple
from datetime import datetime, timedelta

from .base_types import (
    Intent, IntentType, Entity, EntityType,
    Task, ExecutionPlan, DependencyGraph,
    Confidence, ConfidenceLevel
)


class TaskDecomposer:

    def __init__(self, agent_capabilities: Optional[Dict[str, List[str]]] = None, verbose: bool = False):
        self.agent_capabilities = agent_capabilities or {}
        self.verbose = verbose
        self.task_counter = 0

        self.intent_actions = {
            IntentType.CREATE: ['create', 'build', 'generate'],
            IntentType.READ: ['get', 'fetch', 'list', 'search'],
            IntentType.UPDATE: ['update', 'modify', 'change', 'set'],
            IntentType.DELETE: ['delete', 'remove', 'close'],
            IntentType.ANALYZE: ['review', 'analyze', 'check'],
            IntentType.COORDINATE: ['notify', 'send', 'post'],
            IntentType.SEARCH: ['search', 'find', 'query'],
        }

        self.entity_agent_hints = {
            EntityType.ISSUE: ['jira', 'github'],
            EntityType.PR: ['github'],
            EntityType.PROJECT: ['jira'],
            EntityType.CHANNEL: ['slack'],
            EntityType.FILE: ['github', 'browser', 'scraper'],
            EntityType.CODE: ['code_reviewer', 'github'],
        }

    def decompose(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        context: Optional[Dict] = None
    ) -> ExecutionPlan:

        tasks = []

        if len(intents) == 1 and len(entities) <= 2:
            task = self._create_single_task(message, intents[0], entities)
            tasks.append(task)
        else:
            for intent in intents[:3]:
                relevant_entities = [e for e in entities if self._entity_relevant_for_intent(e, intent)]
                task = self._create_task_from_intent(message, intent, relevant_entities)
                tasks.append(task)

        graph = self._build_dependency_graph(tasks, intents, entities)

        plan = ExecutionPlan(
            tasks=tasks,
            dependency_graph=graph,
            estimated_duration=sum(t.estimated_duration for t in tasks),
            estimated_cost=sum(t.estimated_cost for t in tasks)
        )

        if self.verbose:
            print(f"[DECOMPOSITION] Created {len(tasks)} task(s)")
            for task in tasks:
                print(f"  - {task}")

        return plan

    def _create_single_task(self, message: str, intent: Intent, entities: List[Entity]) -> Task:
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        agent = self._select_best_agent(intent, entities)

        action = self._derive_action_from_intent(intent)

        inputs = {
            'message': message,
            'intent': intent.type.value,
            'entities': [{'type': e.type.value, 'value': e.value} for e in entities],
        }

        return Task(
            id=task_id,
            action=action,
            agent=agent,
            inputs=inputs,
            priority=self._calculate_priority(intent, entities),
            estimated_duration=1.0,
            estimated_cost=0.01
        )

    def _create_task_from_intent(self, message: str, intent: Intent, entities: List[Entity]) -> Task:
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"

        agent = self._select_best_agent(intent, entities)

        action = self._derive_action_from_intent(intent)

        inputs = {
            'message': message,
            'intent': intent.type.value,
            'entities': [{'type': e.type.value, 'value': e.value} for e in entities],
        }

        return Task(
            id=task_id,
            action=action,
            agent=agent,
            inputs=inputs,
            priority=self._calculate_priority(intent, entities),
            estimated_duration=1.0,
            estimated_cost=0.01
        )

    def _derive_action_from_intent(self, intent: Intent) -> str:
        action_map = self.intent_actions.get(intent.type, ['execute'])
        return action_map[0]

    def _entity_relevant_for_intent(self, entity: Entity, intent: Intent) -> bool:
        if intent.type == IntentType.CREATE:
            return entity.type in [EntityType.PROJECT, EntityType.ISSUE, EntityType.CHANNEL]
        elif intent.type == IntentType.READ:
            return entity.type in [EntityType.ISSUE, EntityType.PR, EntityType.PROJECT, EntityType.CHANNEL]
        elif intent.type == IntentType.UPDATE:
            return entity.type in [EntityType.ISSUE, EntityType.PR, EntityType.FILE]
        elif intent.type == IntentType.DELETE:
            return entity.type in [EntityType.ISSUE, EntityType.PR, EntityType.CHANNEL]
        elif intent.type == IntentType.SEARCH:
            return entity.type in [EntityType.PROJECT, EntityType.REPOSITORY]
        return True

    def _select_best_agent(self, intent: Intent, entities: List[Entity]) -> Optional[str]:
        if not entities:
            if intent.type == IntentType.SEARCH:
                return 'github'
            return None

        agent_scores: Dict[str, float] = {}

        for entity in entities:
            hints = self.entity_agent_hints.get(entity.type, [])
            for agent in hints:
                agent_scores[agent] = agent_scores.get(agent, 0.0) + entity.confidence

        if agent_scores:
            best_agent = max(agent_scores.items(), key=lambda x: x[1])[0]
            return best_agent

        return None

    def _calculate_priority(self, intent: Intent, entities: List[Entity]) -> int:
        priority = 5

        if intent.type == IntentType.DELETE:
            priority = 1
        elif intent.type == IntentType.UPDATE:
            priority = 2
        elif intent.type == IntentType.CREATE:
            priority = 3
        elif intent.type == IntentType.READ:
            priority = 4

        for entity in entities:
            if entity.type == EntityType.PRIORITY:
                value_lower = entity.value.lower()
                if any(p in value_lower for p in ['p0', 'critical', 'urgent']):
                    priority = min(priority, 1)
                elif any(p in value_lower for p in ['p1', 'high']):
                    priority = min(priority, 2)

        return priority

    def _build_dependency_graph(
        self,
        tasks: List[Task],
        intents: List[Intent],
        entities: List[Entity]
    ) -> DependencyGraph:
        graph = DependencyGraph()

        for task in tasks:
            graph.add_task(task)

        create_tasks = [t for t in tasks if 'create' in t.action.lower()]
        update_tasks = [t for t in tasks if 'update' in t.action.lower()]
        delete_tasks = [t for t in tasks if 'delete' in t.action.lower()]

        for update_task in update_tasks:
            for create_task in create_tasks:
                graph.add_dependency(create_task.id, update_task.id)

        for delete_task in delete_tasks:
            for task in tasks:
                if task != delete_task:
                    graph.add_dependency(task.id, delete_task.id)

        if graph.has_cycle():
            if self.verbose:
                print("[DECOMPOSITION] WARNING: Dependency cycle detected, using sequential order")
            graph = DependencyGraph()
            for task in tasks:
                graph.add_task(task)

        return graph

    def get_metrics(self) -> Dict:
        return {
            'tasks_created': self.task_counter,
            'verbose': self.verbose,
        }

    def reset_metrics(self):
        self.task_counter = 0


class ConfidenceScorer:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def score_overall(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
    ) -> Confidence:

        factors = {}

        if not intents:
            intent_confidence = 0.3
            factors['intent_clarity'] = intent_confidence
        else:
            intent_confidence = max(i.confidence for i in intents)
            factors['intent_clarity'] = intent_confidence

        if len(intents) > 1:
            secondary_confidence = sorted([i.confidence for i in intents], reverse=True)[1] if len(intents) > 1 else 0
            factors['intent_ambiguity'] = 1.0 - (secondary_confidence / max(intent_confidence, 0.01))
        else:
            factors['intent_ambiguity'] = 1.0

        if not entities:
            entity_confidence = 0.4
        else:
            entity_confidence = sum(e.confidence for e in entities) / len(entities)

        factors['entity_clarity'] = entity_confidence

        message_words = message.lower().split()
        if any(word in message_words for word in ['maybe', 'might', 'perhaps', 'not sure', 'think']):
            factors['user_uncertainty'] = 0.5
        else:
            factors['user_uncertainty'] = 1.0

        weights = {
            'intent_clarity': 0.4,
            'intent_ambiguity': 0.2,
            'entity_clarity': 0.3,
            'user_uncertainty': 0.1,
        }

        weighted_score = sum(factors[k] * weights[k] for k in factors)

        if self.verbose:
            print(f"[CONFIDENCE] Factors: {factors}")
            print(f"[CONFIDENCE] Weighted score: {weighted_score:.2f}")

        return Confidence.from_score(weighted_score, factors)

    def get_action_recommendation(self, confidence: Confidence) -> Tuple[str, str]:

        if confidence.score >= 0.85:
            return ('proceed', f'High confidence ({confidence.score:.2f}) - safe to proceed')

        elif confidence.score >= 0.65:
            return ('review', f'Moderate confidence ({confidence.score:.2f}) - review before proceeding')

        else:
            return ('clarify', f'Low confidence ({confidence.score:.2f}) - ask for clarification')

    def suggest_clarifications(self, confidence: Confidence, intents: List[Intent]) -> List[str]:
        suggestions = []

        if 'intent_clarity' in confidence.factors and confidence.factors['intent_clarity'] < 0.6:
            suggestions.append("Could you clarify what you'd like me to do?")

        if 'intent_ambiguity' in confidence.factors and confidence.factors['intent_ambiguity'] < 0.7:
            suggestions.append("I detected multiple possible intents. Which one would you like me to focus on?")

        if 'entity_clarity' in confidence.factors and confidence.factors['entity_clarity'] < 0.5:
            suggestions.append("Could you provide more specific details (project names, issue IDs, etc.)?")

        return suggestions

    def should_ask_confirmation(self, confidence: Confidence, intents: List[Intent]) -> bool:

        if not intents:
            return True

        primary_intent = max(intents, key=lambda i: i.confidence)

        if primary_intent.type in [IntentType.DELETE, IntentType.UPDATE]:
            if confidence.score < 0.9:
                return True

        if confidence.score < 0.5:
            return True

        return False

    def use_decision_theory(
        self,
        confidence: Confidence,
        intents: List[Intent]
    ) -> bool:

        p_correct = confidence.score

        cost_wrong_proceed = 10.0
        cost_wrong_clarify = 2.0

        benefit_right_proceed = 5.0
        benefit_right_clarify = 3.0

        eu_proceed = (p_correct * benefit_right_proceed) + ((1 - p_correct) * -cost_wrong_proceed)
        eu_clarify = (p_correct * benefit_right_clarify) + ((1 - p_correct) * -cost_wrong_clarify)

        if intents and max(intents, key=lambda i: i.confidence).type in [IntentType.DELETE]:
            cost_wrong_proceed *= 2

        message_factors = confidence.factors
        if 'user_uncertainty' in message_factors and message_factors['user_uncertainty'] < 0.7:
            eu_clarify += 1.0

        entropy = -sum(
            i.confidence * math.log2(i.confidence + 0.001)
            for i in intents
        ) if intents else 0

        if entropy > 2.0:
            eu_clarify += 0.2

        should_clarify = eu_clarify > eu_proceed

        if self.verbose:
            print(f"[CONFIDENCE] Decision theory:")
            print(f"  EU(proceed): {eu_proceed:.3f}")
            print(f"  EU(clarify): {eu_clarify:.3f}")
            print(f"  Decision: {'CLARIFY' if should_clarify else 'PROCEED'}")

        return should_clarify

    def get_metrics(self) -> Dict:
        return {
            'verbose': self.verbose,
        }

    def reset_metrics(self):
        pass
