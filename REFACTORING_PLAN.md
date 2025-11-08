# Directory Structure Refactoring Plan

## Current State
- **54 Python files** scattered across 8 directories
- Too many small files (some < 200 lines)
- Related functionality split across multiple files
- Hard to navigate for API conversion

## Target State
- **~25 files** with logical groupings
- Clear separation of concerns
- Easy to convert to web API
- Maintainable and scalable

---

## Compaction Strategy

### 1. **core/** (13 files → 5 files)

#### Keep Separate:
- `observability.py` (389 lines) - Main observability entry point
- `input_validator.py` (148 lines) - Security-critical, keep isolated

#### Merge Into:
**`core/errors.py`** (NEW)
- ← `error_handler.py` (497 lines)
- ← `error_messaging.py` (380 lines)
- **Total: ~880 lines**

**`core/resilience.py`** (NEW)
- ← `retry_manager.py` (332 lines)
- ← `undo_manager.py` (398 lines)
- **Total: ~730 lines**

**`core/user.py`** (NEW)
- ← `analytics.py` (476 lines)
- ← `user_preferences.py` (474 lines)
- ← `message_confirmation.py` (782 lines)
- **Total: ~1732 lines**

#### Keep Internal to Observability:
- `distributed_tracing.py` → stays (used by observability)
- `logging_config.py` → stays (used by observability)
- `orchestration_logger.py` → stays (used by observability)
- `intelligence_logger.py` → stays (used by observability)
- `metrics_aggregator.py` → stays (used by observability)
- `logger.py` → stays (compatibility wrapper)

---

### 2. **intelligence/** (8 files → 3 files)

#### Keep Separate:
- `base_types.py` (664 lines) - Foundation, imported everywhere

#### Merge Into:
**`intelligence/pipeline.py`** (NEW)
- ← `intent_classifier.py` (686 lines)
- ← `entity_extractor.py` (756 lines)
- ← `task_decomposer.py` (481 lines)
- ← `confidence_scorer.py` (728 lines)
- **Total: ~2651 lines** - Main processing pipeline

**`intelligence/system.py`** (NEW)
- ← `context_manager.py` (415 lines)
- ← `cache_layer.py` (353 lines)
- ← `coordinator.py` (529 lines)
- **Total: ~1297 lines** - System infrastructure

---

### 3. **orchestration/** (5 files → 2 files)

**`orchestration/actions.py`** (NEW)
- ← `action_parser.py` (226 lines)
- ← `action_model.py` (215 lines)
- ← `action_enricher.py` (211 lines)
- **Total: ~652 lines**

**`orchestration/confirmation.py`** (NEW)
- ← `confirmation_queue.py` (170 lines)
- **Total: ~170 lines**

---

### 4. **ui/** (5 files → 2 files)

**`ui/terminal.py`** (NEW)
- ← `terminal_ui.py` (305 lines)
- ← `enhanced_terminal_ui.py` (472 lines) *(if still used)*
- **Total: ~777 lines**

**`ui/interactive.py`** (NEW)
- ← `confirmation_ui.py` (235 lines)
- ← `interactive_editor.py` (384 lines)
- **Total: ~619 lines**

---

### 5. **connectors/** (Clean up)

#### Keep (Agent files):
- `base_agent.py`
- `slack_agent.py`
- `jira_agent.py`
- `github_agent.py`
- `notion_agent.py`
- `browser_agent.py`
- `scraper_agent.py`
- `code_reviewer_agent.py`
- `agent_intelligence.py`
- `agent_logger.py`
- `mcp_config.py`
- `tool_manager.py`

#### DELETE (Old/Unused):
- `base_connector.py` (redundant with base_agent.py)
- `jira_connector.py` (old implementation, replaced by jira_agent.py)
- `slack_connector.py` (old implementation, replaced by slack_agent.py)

---

### 6. **llms/** (Keep as is)
- `base_llm.py`
- `gemini_flash.py`

---

### 7. **Root files** (Keep as is)
- `orchestrator.py` (main orchestrator)
- `main.py` (entry point)
- `config.py` (configuration)

---

### 8. **scripts/** (Keep)
- `migrate_logging.py`

---

### 9. **tools/** (Keep)
- `session_viewer.py`

---

## Final Structure

```
.
├── config.py
├── main.py
├── orchestrator.py
├── connectors/
│   ├── base_agent.py
│   ├── *_agent.py (8 agents)
│   ├── agent_intelligence.py
│   ├── agent_logger.py
│   ├── mcp_config.py
│   └── tool_manager.py
├── core/
│   ├── __init__.py
│   ├── observability.py (entry point)
│   ├── distributed_tracing.py
│   ├── logging_config.py
│   ├── orchestration_logger.py
│   ├── intelligence_logger.py
│   ├── metrics_aggregator.py
│   ├── logger.py
│   ├── input_validator.py
│   ├── errors.py ✨ (NEW - merged)
│   ├── resilience.py ✨ (NEW - merged)
│   └── user.py ✨ (NEW - merged)
├── intelligence/
│   ├── __init__.py
│   ├── base_types.py
│   ├── pipeline.py ✨ (NEW - merged)
│   └── system.py ✨ (NEW - merged)
├── llms/
│   ├── base_llm.py
│   └── gemini_flash.py
├── orchestration/
│   ├── __init__.py
│   ├── actions.py ✨ (NEW - merged)
│   └── confirmation.py ✨ (NEW - merged)
├── ui/
│   ├── __init__.py
│   ├── terminal.py ✨ (NEW - merged)
│   └── interactive.py ✨ (NEW - merged)
├── scripts/
│   └── migrate_logging.py
└── tools/
    └── session_viewer.py
```

---

## File Count

**Before:** 54 Python files
**After:** ~28 Python files
**Reduction:** 48%

---

## Benefits

1. **API-Ready**: Clear module boundaries for web API
2. **Less Cognitive Load**: Related code grouped together
3. **Easier Navigation**: Fewer files to search through
4. **Better Imports**: Less import sprawl
5. **Maintainable**: Logical groupings make changes easier

---

## Migration Steps

1. Create new merged files
2. Update imports throughout codebase
3. Test thoroughly
4. Delete old files
5. Update documentation
6. Commit changes

---

## Backward Compatibility

All public APIs will remain accessible through `__init__.py` files with proper imports:

```python
# intelligence/__init__.py
from .pipeline import IntentClassifier, EntityExtractor
from .system import ConversationContextManager, Coordinator
from .base_types import Confidence, Intent, Entity
```

This ensures existing code continues to work.
