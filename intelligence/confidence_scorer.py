"""
Confidence Scoring System

Scores confidence in decisions and determines when to:
- Proceed automatically (high confidence)
- Confirm with user (medium confidence)
- Ask clarifying questions (low confidence)

Author: AI System
Version: 2.0
"""

from typing import List, Dict, Optional, Tuple
from .base_types import (
    Confidence, ConfidenceLevel, Intent, Entity,
    ExecutionPlan, Task
)


class ConfidenceScorer:
    """
    Score confidence in understanding and decisions

    Confidence factors:
    - Intent clarity (how clear is what user wants)
    - Entity completeness (do we have all needed info)
    - Task decomposition quality
    - Agent selection certainty
    - Historical success patterns
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Minimum required entities for each intent type
        self.required_entities = {
            'create': ['issue', 'pr', 'project', 'repository'],  # At least one
            'update': ['issue', 'pr', 'resource'],               # At least one
            'coordinate': ['channel', 'person', 'team'],         # At least one
            'analyze': ['code', 'file', 'repository'],           # At least one
        }

    def score_overall(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        plan: Optional[ExecutionPlan] = None
    ) -> Confidence:
        """
        Score overall confidence in understanding and plan

        Args:
            message: User message
            intents: Detected intents
            entities: Extracted entities
            plan: Execution plan (if created)

        Returns:
            Confidence score with factors
        """
        factors = {}

        # Factor 1: Intent clarity (0-1.0)
        factors['intent_clarity'] = self._score_intent_clarity(message, intents)

        # Factor 2: Entity completeness (0-1.0)
        factors['entity_completeness'] = self._score_entity_completeness(intents, entities)

        # Factor 3: Message ambiguity (0-1.0, lower is more ambiguous)
        factors['message_clarity'] = self._score_message_clarity(message)

        # Factor 4: Plan quality (if available)
        if plan:
            factors['plan_quality'] = self._score_plan_quality(plan)
        else:
            factors['plan_quality'] = 0.5

        # Calculate weighted average
        weights = {
            'intent_clarity': 0.3,
            'entity_completeness': 0.3,
            'message_clarity': 0.2,
            'plan_quality': 0.2
        }

        total_score = sum(factors[k] * weights[k] for k in factors)

        # Identify uncertainties
        uncertainties = self._identify_uncertainties(message, intents, entities, factors)

        # Identify assumptions
        assumptions = self._identify_assumptions(message, intents, entities)

        confidence = Confidence.from_score(total_score, factors)
        confidence.uncertainties = uncertainties
        confidence.assumptions = assumptions

        if self.verbose:
            print(f"[CONFIDENCE] Overall score: {confidence}")
            print(f"  Factors: {factors}")
            if uncertainties:
                print(f"  Uncertainties: {len(uncertainties)}")
            if assumptions:
                print(f"  Assumptions: {len(assumptions)}")

        return confidence

    def _score_intent_clarity(self, message: str, intents: List[Intent]) -> float:
        """
        Score how clear the user's intent is

        Factors:
        - Number of intents (1 is clearest)
        - Intent confidence scores
        - Presence of action words
        """
        if not intents:
            return 0.2  # Very unclear

        # High confidence intents
        high_conf_intents = [i for i in intents if i.confidence > 0.8]
        if not high_conf_intents:
            return 0.4  # Low confidence in all intents

        # Single clear intent is best
        if len(high_conf_intents) == 1:
            return min(high_conf_intents[0].confidence, 0.95)

        # Multiple clear intents is okay but slightly lower confidence
        if len(high_conf_intents) <= 3:
            avg_confidence = sum(i.confidence for i in high_conf_intents) / len(high_conf_intents)
            return avg_confidence * 0.9

        # Too many intents - might be unclear
        return 0.6

    def _score_entity_completeness(self, intents: List[Intent], entities: List[Entity]) -> float:
        """
        Score whether we have all needed entities for the intents

        Checks if we have the minimum required entities
        """
        if not intents:
            return 0.0

        primary_intent = intents[0]
        intent_type = primary_intent.type.value

        # Check if we have required entities
        required = self.required_entities.get(intent_type, [])

        if not required:
            # No specific requirements
            return 0.8

        # Check if we have at least one required entity
        entity_types = [e.type.value for e in entities]
        has_required = any(req in entity_types for req in required)

        if has_required:
            # Have at least one required entity
            # Score based on number of high-confidence entities
            high_conf_entities = [e for e in entities if e.confidence > 0.8]
            if len(high_conf_entities) >= 2:
                return 0.95
            elif len(high_conf_entities) == 1:
                return 0.80
            else:
                return 0.60
        else:
            # Missing required entities
            return 0.3

    def _score_message_clarity(self, message: str) -> float:
        """
        Score clarity of message itself

        Factors:
        - Length (too short or too long is unclear)
        - Question words (many questions = seeking info)
        - Specificity (specific details = clearer)
        """
        message_lower = message.lower()
        words = message_lower.split()
        word_count = len(words)

        score = 0.5  # Base score

        # Length scoring
        if 5 <= word_count <= 30:
            score += 0.2  # Good length
        elif word_count < 5:
            score -= 0.2  # Too short
        elif word_count > 50:
            score -= 0.1  # Too long

        # Question words reduce clarity for action requests
        question_words = ['what', 'how', 'why', 'when', 'where', 'which']
        question_count = sum(1 for qw in question_words if qw in message_lower)
        if question_count > 2:
            score -= 0.2  # Many questions = uncertain

        # Specific details increase clarity
        specific_indicators = ['#', '@', '-', '/', 'http']
        specificity = sum(1 for ind in specific_indicators if ind in message)
        score += min(specificity * 0.05, 0.2)

        return max(0.0, min(1.0, score))

    def _score_plan_quality(self, plan: ExecutionPlan) -> float:
        """
        Score quality of execution plan

        Factors:
        - No circular dependencies
        - Reasonable task count
        - Clear agent assignments
        - No high risks
        """
        score = 0.8  # Base score

        # Check for critical risks
        if plan.risks:
            critical_risks = [r for r in plan.risks if 'CRITICAL' in r]
            if critical_risks:
                score -= 0.5  # Major issue
            else:
                score -= 0.1 * len(plan.risks)

        # Check agent assignments
        tasks_without_agents = [t for t in plan.tasks if not t.agent]
        if tasks_without_agents:
            score -= 0.1 * len(tasks_without_agents) / len(plan.tasks)

        # Check task count
        if len(plan.tasks) == 0:
            score = 0.0
        elif len(plan.tasks) > 15:
            score -= 0.1  # Too many tasks

        return max(0.0, min(1.0, score))

    def _identify_uncertainties(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity],
        factors: Dict[str, float]
    ) -> List[str]:
        """
        Identify specific uncertainties that should be clarified

        Returns list of uncertainty descriptions
        """
        uncertainties = []

        # Low intent clarity
        if factors.get('intent_clarity', 0) < 0.6:
            uncertainties.append("Unclear what action to take")

        # Low entity completeness
        if factors.get('entity_completeness', 0) < 0.6:
            if intents:
                primary_intent = intents[0].type.value
                uncertainties.append(f"Missing information for {primary_intent} action")

        # Ambiguous references
        ambiguous_words = ['it', 'that', 'this', 'them', 'those']
        message_lower = message.lower()
        has_ambiguous = any(word in message_lower.split() for word in ambiguous_words)

        if has_ambiguous and len(entities) == 0:
            uncertainties.append("Ambiguous references without context")

        # Multiple high-confidence intents
        if intents:
            high_conf = [i for i in intents if i.confidence > 0.7]
            if len(high_conf) > 3:
                uncertainties.append(f"Multiple actions requested ({len(high_conf)})")

        # No project/resource specified for create/update
        if intents and intents[0].type.value in ['create', 'update']:
            has_project = any(e.type.value in ['project', 'repository'] for e in entities)
            if not has_project:
                uncertainties.append("Project/repository not specified")

        return uncertainties

    def _identify_assumptions(
        self,
        message: str,
        intents: List[Intent],
        entities: List[Entity]
    ) -> List[str]:
        """
        Identify assumptions we're making

        Important to communicate these to user
        """
        assumptions = []

        # Assumption about which project
        if intents and intents[0].type.value in ['create', 'update']:
            has_explicit_project = any(e.type.value in ['project', 'repository'] for e in entities)
            if not has_explicit_project:
                assumptions.append("Using current/default project")

        # Assumption about priority
        has_priority = any(e.type.value == 'priority' for e in entities)
        if not has_priority and intents and intents[0].type.value == 'create':
            assumptions.append("Using default priority (medium)")

        # Assumption about assignee
        has_assignee = any(e.type.value == 'person' for e in entities)
        if not has_assignee and intents and intents[0].type.value == 'create':
            assumptions.append("Leaving unassigned")

        return assumptions

    def suggest_clarifications(self, confidence: Confidence, intents: List[Intent]) -> List[str]:
        """
        Suggest what clarifying questions to ask

        Args:
            confidence: Confidence score
            intents: Detected intents

        Returns:
            List of clarifying questions to ask user
        """
        questions = []

        # Ask based on uncertainties
        if "Unclear what action" in str(confidence.uncertainties):
            questions.append("What would you like me to do?")

        if "Missing information" in str(confidence.uncertainties):
            if intents:
                primary = intents[0].type.value
                if primary == 'create':
                    questions.append("What should I create? (issue, PR, page, etc.)")
                elif primary == 'update':
                    questions.append("What should I update?")
                elif primary == 'coordinate':
                    questions.append("Who should I notify?")

        if "Project/repository not specified" in str(confidence.uncertainties):
            questions.append("Which project or repository?")

        if "Ambiguous references" in str(confidence.uncertainties):
            questions.append("Can you clarify what 'it' or 'that' refers to?")

        return questions

    def should_proceed_automatically(self, confidence: Confidence) -> bool:
        """Should we proceed without asking user?"""
        return confidence.should_proceed()

    def should_confirm_with_user(self, confidence: Confidence) -> bool:
        """Should we confirm plan with user before executing?"""
        return confidence.should_confirm()

    def should_ask_clarifying_questions(self, confidence: Confidence) -> bool:
        """Should we ask clarifying questions?"""
        return confidence.should_clarify()

    def get_action_recommendation(self, confidence: Confidence) -> Tuple[str, str]:
        """
        Get recommended action based on confidence

        Returns:
            (action, explanation) tuple
            action: 'proceed', 'confirm', or 'clarify'
        """
        if self.should_proceed_automatically(confidence):
            return ('proceed', f"High confidence ({confidence.score:.2f}) - proceeding automatically")

        elif self.should_confirm_with_user(confidence):
            return ('confirm', f"Medium confidence ({confidence.score:.2f}) - confirming plan with user")

        else:
            return ('clarify', f"Low confidence ({confidence.score:.2f}) - asking clarifying questions")
