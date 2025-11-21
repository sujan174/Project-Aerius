# Project-Friday Intelligence System - Documentation Index

This directory contains comprehensive documentation of the Project-Friday intelligence system.

## Documents

### 1. **INTELLIGENCE_SUMMARY.md** - START HERE
Quick reference guide for understanding the intelligence system.
- What is the intelligence system
- Key components overview
- Quick answers to common questions
- Performance metrics
- Troubleshooting guide
- Developer tips

**Best for:** Getting a quick understanding, troubleshooting, development quick reference

---

### 2. **INTELLIGENCE_SYSTEM.md** - COMPREHENSIVE GUIDE
Deep, detailed analysis of the entire intelligence system (641 lines).

Covers:
- **Section 1**: Core architecture - Hybrid Intelligence System v5.0
- **Section 2**: Intelligence flow and detailed component descriptions
- **Section 3**: Memory and context systems
- **Section 4**: Entity extraction and relationships
- **Section 5**: Task decomposition and execution planning
- **Section 6**: Confidence scoring and decision theory
- **Section 7**: Safety and reliability systems (circuit breaker, etc.)
- **Section 8**: Learning and adaptation mechanisms
- **Section 9**: Performance metrics and monitoring
- **Section 10**: Key differentiators and why this works
- **Section 11**: Integration with orchestrator
- **Section 12**: Example flow walkthrough
- **Section 13**: Conclusion

**Best for:** Complete understanding, system design review, implementation reference

---

### 3. **ARCHITECTURE_DIAGRAM.txt** - VISUAL OVERVIEW
ASCII art diagrams and visual representations of the system.

Includes:
- System architecture overview
- Data flow example
- Component interaction matrix
- Metrics and monitoring dashboard
- Production features summary

**Best for:** Visual learners, presentations, system design documentation

---

## Quick Navigation

### Understanding the System
1. Start with `INTELLIGENCE_SUMMARY.md` (10 minutes read)
2. Review `ARCHITECTURE_DIAGRAM.txt` (5 minutes visual)
3. Deep dive into `INTELLIGENCE_SYSTEM.md` if needed

### Key Questions Answered

**"What is the core intelligence?"**
→ See `INTELLIGENCE_SUMMARY.md` Section 1

**"How does it work?"**
→ See `ARCHITECTURE_DIAGRAM.txt` Data Flow Example

**"What are the components?"**
→ See `INTELLIGENCE_SUMMARY.md` Section 3 or `INTELLIGENCE_SYSTEM.md` Section 1.2

**"How does memory work?"**
→ See `INTELLIGENCE_SUMMARY.md` Section 5 or `INTELLIGENCE_SYSTEM.md` Section 3

**"What about learning?"**
→ See `INTELLIGENCE_SUMMARY.md` Section 6 or `INTELLIGENCE_SYSTEM.md` Section 8

**"Safety systems?"**
→ See `INTELLIGENCE_SUMMARY.md` Section 7 or `INTELLIGENCE_SYSTEM.md` Section 7

**"How to debug issues?"**
→ See `INTELLIGENCE_SUMMARY.md` Troubleshooting section

**"How to add new features?"**
→ See `INTELLIGENCE_SUMMARY.md` For Developers section

---

## The Intelligence System at a Glance

### Architecture
```
Hybrid Intelligence System v5.0
    ├─ Tier 1: Fast Keyword Filter (10ms, free)
    └─ Tier 2: LLM Classifier (200ms, semantic)

Support Systems
    ├─ Intent Classifier
    ├─ Entity Extractor
    ├─ Task Decomposer
    ├─ Confidence Scorer
    ├─ Context Manager
    ├─ Cache Layer
    ├─ Circuit Breaker
    └─ Error Handler
```

### Key Performance Metrics
- **Latency**: 80ms average
- **Accuracy**: 92%
- **Cost**: $0.0065/1K requests
- **Cache Hit Rate**: 70-80%

### Core Capabilities
1. Intent classification (what user wants)
2. Entity extraction (objects and relationships)
3. Task decomposition (break into steps)
4. Confidence scoring (how sure are we)
5. Context tracking (remember history)
6. Pattern learning (improve over time)
7. Safety management (prevent failures)

---

## File Structure

```
project-root/
├── intelligence/
│   ├── __init__.py
│   ├── base_types.py              # Data structures
│   ├── cache_layer.py             # LRU cache with TTL
│   ├── confidence_scorer.py        # Bayesian confidence
│   ├── context_manager.py          # Conversation tracking
│   ├── entity_extractor.py         # Entity extraction
│   ├── fast_filter.py              # Tier 1 keyword matching
│   ├── hybrid_system.py            # Two-tier orchestrator
│   ├── intent_classifier.py        # Intent detection
│   ├── llm_classifier.py           # Tier 2 semantic analysis
│   └── task_decomposer.py          # Task planning
│
├── core/
│   ├── circuit_breaker.py          # Failure prevention
│   ├── error_handler.py            # Error management
│   ├── retry_manager.py            # Smart retries
│   ├── user_preferences.py         # Learning system
│   └── ... (other core systems)
│
├── orchestrator.py                  # Main integration
├── main.py                          # Entry point
│
├── INTELLIGENCE_SYSTEM.md           # (This package)
├── INTELLIGENCE_SUMMARY.md
├── ARCHITECTURE_DIAGRAM.txt
└── DOCUMENTATION_INDEX.md           # (You are here)
```

---

## Component Deep Dives

### Hybrid Intelligence System (`intelligence/hybrid_system.py`)
- **Purpose**: Two-tier routing for optimal speed/accuracy balance
- **Tier 1**: Fast keyword filter (~10ms)
- **Tier 2**: LLM semantic analysis (~200ms)
- **Smart Routing**: Routes based on confidence scores
- **Metrics**: Tracks performance of both paths

### Fast Keyword Filter (`intelligence/fast_filter.py`)
- **Purpose**: Ultra-fast intent detection for obvious requests
- **Method**: Pre-compiled regex patterns
- **Coverage**: 35-40% of requests
- **Accuracy**: 95% for covered patterns
- **Performance**: <10ms latency

### LLM Classifier (`intelligence/llm_classifier.py`)
- **Purpose**: Semantic understanding for complex requests
- **Method**: Gemini Flash with semantic caching
- **Coverage**: 60-65% of requests
- **Caching**: 70-80% hit rate
- **Accuracy**: 92%

### Intent Classifier (`intelligence/intent_classifier.py`)
- **Intents**: CREATE, READ, UPDATE, DELETE, ANALYZE, COORDINATE, SEARCH, WORKFLOW
- **Features**: Multi-intent detection, implicit requirements
- **Calibration**: Adjusts confidence based on message clarity

### Entity Extractor (`intelligence/entity_extractor.py`)
- **Entity Types**: 15+ types (Issues, PRs, Projects, People, Teams, Channels, etc.)
- **Methods**: Regex patterns, relationship extraction, normalization
- **Features**: Coreference resolution, confidence calibration

### Task Decomposer (`intelligence/task_decomposer.py`)
- **Tasks**: Converts intents to executable tasks
- **Graphs**: Builds dependency graphs for execution order
- **Planning**: Estimates cost/duration, identifies risks
- **Optimization**: Identifies parallelizable tasks

### Confidence Scorer (`intelligence/confidence_scorer.py`)
- **Algorithm**: Bayesian estimation with multi-factor scoring
- **Factors**: Intent clarity, entity completeness, message clarity, plan quality
- **Decision**: PROCEED (>0.8), CONFIRM (0.6-0.8), CLARIFY (<0.6)
- **Advanced**: Entropy calculation, decision theory

### Context Manager (`intelligence/context_manager.py`)
- **History**: Maintains conversation turns
- **Entities**: Tracks across turns with mention counts
- **References**: Resolves pronouns and type-specific references
- **Learning**: Learns user patterns

### Cache Layer (`intelligence/cache_layer.py`)
- **Policy**: LRU (Least Recently Used) eviction
- **TTL**: 5 minutes default with customization
- **Thread-Safe**: Safe for concurrent access
- **Statistics**: Tracks hits, misses, evictions

---

## How to Use This Documentation

### For Quick Understanding
1. Read `INTELLIGENCE_SUMMARY.md` (15 minutes)
2. Reference `ARCHITECTURE_DIAGRAM.txt` for visuals

### For Implementation
1. Start with `INTELLIGENCE_SYSTEM.md` Section 1-2 (architecture)
2. Read relevant component section
3. Check code in `/intelligence/` directory

### For Debugging
1. Check `INTELLIGENCE_SUMMARY.md` troubleshooting
2. Enable verbose mode on components
3. Check logs and metrics

### For Development
1. Read "For Developers" section in `INTELLIGENCE_SUMMARY.md`
2. Understand component in `INTELLIGENCE_SYSTEM.md`
3. Review code and add your changes

---

## Key Concepts Explained

### Confidence Score
A probability (0.0-1.0) indicating how sure the system is about its understanding.
- Combines: intent clarity, entity completeness, message clarity, plan quality
- Used for: deciding whether to proceed, confirm, or clarify
- Updated: through pattern learning and calibration

### Hybrid Architecture
Two-tier system balancing speed and accuracy:
- **Tier 1 (Fast)**: For obvious requests (10ms, free)
- **Tier 2 (Semantic)**: For complex requests (200ms, paid)
- **Smart Routing**: Uses confidence to decide which to use

### Semantic Caching
Caching strategy that dramatically reduces LLM calls:
- **Hit Rate**: 70-80% for cached requests
- **Speedup**: From 200ms to 20ms on hit
- **Cost**: Reduces API costs by 70-80%

### Circuit Breaker
Reliability pattern preventing cascading failures:
- **CLOSED**: Normal operation
- **OPEN**: After threshold failures, blocks requests
- **HALF_OPEN**: Testing recovery
- **Auto-Recovery**: After timeout period

### Pattern Learning
System learns from user behavior:
- **Patterns**: Recurring action sequences
- **Success Tracking**: Counts successes per pattern
- **Confidence**: Increases with successful patterns

---

## Performance Targets

### Hybrid System v5.0
- **Latency**: 80ms average (target achieved)
- **Accuracy**: 92% (target achieved)
- **Cost**: $0.0065/1K requests (target achieved)
- **Fast Path**: 35-40% coverage (target achieved)
- **LLM Path**: 60-65% coverage (target achieved)

---

## Common Tasks

### Debug Low Confidence
1. Check intent clarity: Is request ambiguous?
2. Check entity completeness: Are all entities specified?
3. Check message clarity: Is message well-formed?
4. Check plan quality: Are there circular dependencies?

### Improve Response Time
1. Enable caching: Check cache hit rate (should be 70%+)
2. Use fast path: Make requests simpler and more specific
3. Monitor circuit breaker: Check agent health

### Add New Intent Type
1. Add to `IntentType` enum in `base_types.py`
2. Add keywords to `FastKeywordFilter.PATTERNS`
3. Update prompts in `LLMIntentClassifier`

### Reduce Costs
1. Enable semantic caching
2. Route simple requests to fast path
3. Batch similar requests together

---

## Related Documentation

- **Orchestrator Integration**: See `orchestrator.py`
- **Sub-agents**: See `/connectors/` directory
- **Safety Systems**: See `/core/` directory
- **Observability**: See observability module

---

## Questions?

Refer to the appropriate document:
1. **Quick questions**: `INTELLIGENCE_SUMMARY.md`
2. **How it works**: `ARCHITECTURE_DIAGRAM.txt` + `INTELLIGENCE_SYSTEM.md`
3. **Implementation**: Component-specific sections in `INTELLIGENCE_SYSTEM.md`
4. **Troubleshooting**: `INTELLIGENCE_SUMMARY.md` troubleshooting section

---

## Document Versions

| Document | Sections | Pages (Approx) | Version | Last Updated |
|----------|----------|----------------|---------|--------------|
| INTELLIGENCE_SUMMARY.md | 15 sections | ~20 | 5.0 | 2024 |
| INTELLIGENCE_SYSTEM.md | 13 sections | ~25 | 5.0 | 2024 |
| ARCHITECTURE_DIAGRAM.txt | 5 sections | ~8 | 5.0 | 2024 |
| DOCUMENTATION_INDEX.md | This doc | ~10 | 5.0 | 2024 |

---

## System Status

- **Status**: Production-Ready
- **Version**: 5.0 (Hybrid Intelligence System)
- **Coverage**: 92% accuracy, 80ms latency
- **Safety**: Circuit breaker, error handling, retry management
- **Learning**: Pattern learning, confidence calibration, entity adaptation
- **Monitoring**: Full observability with metrics and tracing

---

**Navigation**: You are reading DOCUMENTATION_INDEX.md
Start with: `INTELLIGENCE_SUMMARY.md` for quick understanding
Deep dive: `INTELLIGENCE_SYSTEM.md` for comprehensive details
Visual: `ARCHITECTURE_DIAGRAM.txt` for system design

