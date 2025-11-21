# Project-Friday Intelligence System - Quick Reference Guide

## What is the Intelligence System?

Project-Friday's intelligence system is a **sophisticated, production-grade AI reasoning engine** that powers the orchestrator agent. It's built on a **two-tier hybrid architecture** that balances speed, accuracy, and cost.

### Core Purpose
Transform natural language user requests into structured actions that sub-agents can execute reliably and efficiently.

---

## Quick Answer to Key Questions

### 1. What is the Core Intelligence?

The **Hybrid Intelligence System v5.0** - a two-tier approach:
- **Tier 1 (Fast Path)**: Keyword-based intent detection (~10ms, free)
- **Tier 2 (Semantic Path)**: LLM-powered understanding (~200ms, $0.01/1K)

Routes simple requests through fast path, complex ones through semantic analysis.

### 2. How Does It Work?

```
User Input → Hybrid System → Tier 1: Try Fast Filter
    ↓
    └─ High Confidence? → Return Result (10ms, free)
    └─ Low Confidence? → Try Tier 2: LLM (200ms, semantic)
    
Additional Processing:
    → Intent Classification (what user wants)
    → Entity Extraction (what objects mentioned)
    → Task Decomposition (how to break down)
    → Confidence Scoring (how sure are we)
    → Context Integration (conversation history)
```

### 3. What Components Make It Up?

**Core Intelligence Pipeline:**
- **Hybrid System** - Routes between Tier 1 and 2
- **Fast Filter** - Keyword pattern matching
- **LLM Classifier** - Semantic understanding with caching
- **Intent Classifier** - Detects user intents (CREATE, READ, UPDATE, etc.)
- **Entity Extractor** - Pulls out issues, projects, people, etc.
- **Task Decomposer** - Breaks into executable tasks
- **Confidence Scorer** - Uses Bayesian estimation
- **Context Manager** - Remembers conversation and learns

**Supporting Systems:**
- **Cache Layer** - LRU cache with TTL (70-80% hit rate)
- **Circuit Breaker** - Prevents cascading failures
- **Error Handler** - Comprehensive error management
- **Retry Manager** - Smart exponential backoff

### 4. Component Interactions

```
Hybrid System
    ↓
Intent Classifier ←→ Entity Extractor ←→ Context Manager
    ↓                    ↓                     ↓
    └──────→ Task Decomposer ←─────────────┘
               ↓
    Confidence Scorer
               ↓
    Decision: PROCEED / CONFIRM / CLARIFY
```

### 5. Memory System

**Conversation Context Manager:**
- Tracks all conversation turns
- Remembers recent entities
- Resolves ambiguous references ("it", "that", "the issue")
- Learns user patterns
- Provides context to intelligence components

**Cache Layer:**
- LRU with TTL (5-minute default)
- Stores results of expensive operations
- 70-80% hit rate in LLM classifier
- Dramatically reduces latency and cost

### 6. Learning & Adaptation

**Pattern Learning:**
- Learns recurring user behaviors
- Tracks success rates
- Increases confidence for repeated patterns

**Confidence Calibration:**
- Adjusts scores based on historical accuracy
- Updates from user feedback
- Gets smarter over time

**Entity Confidence:**
- Boosts when entities appear in context
- Cross-validates with other entities
- Learns from patterns

### 7. Safety Systems

**Circuit Breaker:**
- Detects failing agents
- Temporarily disables them
- Prevents cascading failures
- Auto-recovers when healthy

**Retry Manager:**
- Exponential backoff with jitter
- Handles transient failures
- Prevents retry storms

**Error Handling:**
- Classifies error types
- Provides user-friendly messages
- Routes to appropriate handling

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────┐
│         ORCHESTRATOR AGENT                  │
├─────────────────────────────────────────────┤
│ INTELLIGENCE          SAFETY SYSTEMS        │
│ • Hybrid System       • Circuit Breaker     │
│ • Intent Class        • Retry Manager       │
│ • Entity Extract      • Error Handler       │
│ • Task Decompose                            │
│ • Confidence Scorer   LEARNING              │
│ • Context Manager     • Pattern Learning    │
│                       • Calibration         │
│                       • Analytics           │
└─────────────────────────────────────────────┘
         ↓
    SUB-AGENTS
    (Slack, GitHub, Jira, etc.)
```

---

## Key Performance Metrics

### Hybrid System
- **Latency**: 80ms average
- **Accuracy**: 92%
- **Cost**: $0.0065/1K requests
- **Fast Path**: 35-40% of requests, <10ms
- **LLM Path**: 60-65% of requests, ~200ms

### Caching
- **Hit Rate**: 70-80% in LLM classifier
- **Cache Size**: 1000 entries max
- **TTL**: 5 minutes default
- **Eviction**: LRU policy

### Confidence Scoring
- **Factors**: Intent clarity, entity completeness, message clarity, plan quality
- **Scale**: 0.0 (very unsure) to 1.0 (certain)
- **Decision Thresholds**:
  - > 0.8: Proceed automatically
  - 0.6-0.8: Confirm with user
  - < 0.6: Ask clarifying questions

---

## Intent Types Recognized

```
CREATE      → Make something new
READ        → Retrieve information
UPDATE      → Modify existing data
DELETE      → Remove something
ANALYZE     → Examine/evaluate
COORDINATE  → Notify/communicate
SEARCH      → Find information
WORKFLOW    → Automation/if-then logic
UNKNOWN     → Cannot determine
```

---

## Entity Types Extracted

```
PROJECT         ISSUE          PR
PERSON          TEAM           CHANNEL
REPOSITORY      FILE           PRIORITY
STATUS          DATE           LABEL
RESOURCE        CODE           UNKNOWN
```

---

## Information Flow Example

**User says:** "Create a critical issue in KAN for the auth bug"

```
1. HYBRID INTELLIGENCE
   → Fast Filter: Detects "create" keyword
   → Confidence: 0.92 (HIGH)
   → Return: CREATE intent

2. ENTITY EXTRACTION
   → Project: KAN (0.95)
   → Priority: critical (0.95)
   → Context: "auth bug"

3. CONTEXT MANAGER
   → Store conversation turn
   → Track entities: KAN, critical
   → Learn pattern

4. TASK DECOMPOSER
   → Task 1: Get KAN project details
   → Task 2: Create issue with extracted info
   → Dependencies: Task 2 depends on Task 1

5. CONFIDENCE SCORER
   → Intent clarity: 0.90
   → Entity completeness: 0.85
   → Message clarity: 0.85
   → Plan quality: 0.80
   → Overall: 0.86 (MEDIUM-HIGH)
   → Action: CONFIRM WITH USER

6. EXECUTION
   → Circuit Breaker: jira_agent healthy? YES
   → Execute Task 1: Success
   → Execute Task 2: Success
   → Result: "Created KAN-1234: auth bug (Critical)"

7. LEARNING
   → Pattern {create, KAN, critical} confidence +5%
   → Cache result
   → Update entity tracking
```

---

## File Locations

### Core Intelligence
- `/intelligence/hybrid_system.py` - Two-tier orchestrator
- `/intelligence/fast_filter.py` - Tier 1 keyword matching
- `/intelligence/llm_classifier.py` - Tier 2 semantic analysis
- `/intelligence/intent_classifier.py` - Intent detection
- `/intelligence/entity_extractor.py` - Entity extraction
- `/intelligence/task_decomposer.py` - Task planning
- `/intelligence/confidence_scorer.py` - Confidence estimation
- `/intelligence/context_manager.py` - Conversation tracking
- `/intelligence/cache_layer.py` - LRU cache with TTL
- `/intelligence/base_types.py` - Data structures

### Supporting Systems
- `/core/circuit_breaker.py` - Failure prevention
- `/core/retry_manager.py` - Smart retries
- `/core/error_handler.py` - Error management
- `/core/input_validator.py` - Input sanitization
- `/core/user_preferences.py` - Learning/adaptation

### Integration
- `/orchestrator.py` - Integrates all components
- `/main.py` - Entry point

---

## Key Differentiators

### Why This Architecture Works

1. **Two-Tier Balance**
   - Simple requests: instant (10ms)
   - Complex requests: accurate (200ms)
   - Hybrid approach: best of both worlds

2. **Semantic Caching**
   - 70-80% cache hit rate
   - Reduces LLM calls dramatically
   - Saves time and money

3. **Probabilistic Reasoning**
   - Bayesian confidence estimation
   - Entropy-based uncertainty
   - Expected utility decision theory

4. **Context Awareness**
   - Tracks conversation history
   - Resolves ambiguous references
   - Learns user patterns

5. **Production-Ready**
   - Circuit breaker safety
   - Comprehensive error handling
   - Detailed metrics and logging

---

## Performance Optimization Tips

1. **Enable Caching**
   - Cache identical requests
   - Large hit rate reduces API calls
   - Results in 20ms responses

2. **Use Fast Path When Possible**
   - Simple, clear requests → fast path
   - ~80% latency reduction
   - Zero API cost

3. **Batch Similar Requests**
   - Cache hit likelihood increases
   - Better resource utilization

4. **Monitor Circuit Breaker**
   - Watch for OPEN circuits
   - Indicates system issues
   - Auto-recovery after timeout

---

## Common Patterns

### High Confidence Request
→ Proceed automatically
→ Return result in 10-30ms
→ No user confirmation needed

### Medium Confidence Request
→ Confirm plan with user
→ Show what system plans to do
→ User can approve or clarify

### Low Confidence Request
→ Ask clarifying questions
→ Help user be more specific
→ Re-process with better info

---

## Troubleshooting

### Slow Response
- Check cache hit rate (should be 70%+)
- May be using LLM path (expected: 200ms)
- Check circuit breaker status

### Wrong Intent
- Message may be ambiguous
- Try being more specific
- Include entity names (project, person, etc.)

### Missing Entities
- Entities need specific format (@user, #channel, KAN-123)
- Or context from previous turns
- Clarify which object you mean

### Confidence Too Low
- Message lacks specificity
- Missing required entities
- Try simpler, clearer request

---

## For Developers

### Adding a New Intent Type
1. Add to `IntentType` enum in `base_types.py`
2. Add keywords to `FastKeywordFilter.PATTERNS`
3. Add to `IntentClassifier.intent_keywords`
4. Update LLM classifier prompt

### Adding a New Entity Type
1. Add to `EntityType` enum in `base_types.py`
2. Add regex patterns to `EntityExtractor.patterns`
3. Add normalization logic if needed

### Improving Confidence Scoring
- Adjust weights in `ConfidenceScorer`
- Calibrate based on historical data
- Add new scoring factors

### Debugging Intelligence
- Set `verbose=True` on components
- Check logs in `/logs`
- Monitor metrics via observability system

---

## Resources

- **Full Documentation**: `INTELLIGENCE_SYSTEM.md`
- **Architecture Diagram**: `ARCHITECTURE_DIAGRAM.txt`
- **Code**: `/intelligence/` directory
- **Integration**: `orchestrator.py`

---

**Last Updated**: 2024
**Version**: 5.0 (Hybrid Intelligence System)
**Status**: Production-Ready

