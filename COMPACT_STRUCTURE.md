# Proposed Compact Directory Structure

## Overview
Reduce from 54 files to ~28 files while maintaining all functionality.

## Final Structure
```
lazy-devs-orchestrator/
├── config.py
├── main.py  
├── orchestrator.py
│
├── connectors/              # Agent implementations (keep separate)
│   ├── base_agent.py
│   ├── slack_agent.py
│   ├── jira_agent.py
│   ├── github_agent.py
│   ├── notion_agent.py
│   ├── browser_agent.py
│   ├── scraper_agent.py
│   ├── code_reviewer_agent.py
│   ├── agent_intelligence.py
│   ├── agent_logger.py
│   ├── mcp_config.py
│   └── tool_manager.py
│
├── core/                    # System utilities
│   ├── __init__.py
│   ├── observability.py     # Main observability API
│   ├── distributed_tracing.py
│   ├── logging_config.py
│   ├── orchestration_logger.py
│   ├── intelligence_logger.py
│   ├── metrics_aggregator.py
│   ├── logger.py            # Compatibility wrapper
│   ├── input_validator.py   # Security
│   ├── errors.py            # ← error_handler + error_messaging
│   ├── resilience.py        # ← retry_manager + undo_manager
│   └── user.py              # ← analytics + user_preferences + message_confirmation
│
├── intelligence/            # AI intelligence system
│   ├── __init__.py
│   ├── base_types.py        # Foundation types
│   ├── pipeline.py          # ← intent + entity + task + confidence
│   └── system.py            # ← context + cache + coordinator
│
├── llms/                    # LLM abstractions
│   ├── base_llm.py
│   └── gemini_flash.py
│
├── orchestration/           # Action orchestration
│   ├── __init__.py
│   ├── actions.py           # ← parser + model + enricher
│   └── confirmation.py      # ← confirmation_queue
│
├── ui/                      # Terminal UI
│   ├── __init__.py
│   ├── terminal.py          # ← terminal_ui + enhanced_terminal_ui
│   └── interactive.py       # ← confirmation_ui + interactive_editor
│
├── scripts/                 # Utility scripts
│   └── migrate_logging.py
│
├── tools/                   # Developer tools
│   └── session_viewer.py
│
├── docs/                    # Documentation
│   ├── LOGGING_SYSTEM_README.md
│   ├── LOGGING_GUIDE.md
│   ├── SESSION_VIEWER_GUIDE.md
│   └── ...
│
└── logs/                    # Log output (generated)
    ├── orchestration/
    ├── intelligence/
    ├── metrics/
    └── traces/
```

## File Reductions

| Directory | Before | After | Reduction |
|-----------|--------|-------|-----------|
| core/     | 13     | 11    | 15%       |
| intelligence/ | 8  | 3     | 63%       |
| orchestration/ | 5 | 2     | 60%       |
| ui/       | 5      | 2     | 60%       |
| connectors/ | 15   | 12    | 20%       |
| **TOTAL** | **54** | **~28** | **48%** |

## Import Examples (After Refactoring)

```python
# Old way (still works via __init__.py)
from intelligence import IntentClassifier, EntityExtractor

# New way (more explicit)
from intelligence.pipeline import IntentClassifier, EntityExtractor
from intelligence.system import Coordinator, ConversationContextManager

# Core utilities
from core.errors import ErrorClassifier, ErrorMessageEnhancer
from core.resilience import RetryManager, UndoManager
from core.user import AnalyticsCollector, UserPreferenceManager

# Orchestration
from orchestration.actions import ActionParser, ActionEnricher
from orchestration.confirmation import ConfirmationQueue
```

## Benefits for Web API

When converting to FastAPI:

```python
# api/routes/orchestration.py
from orchestrator import OrchestratorAgent
from core.user import AnalyticsCollector
from intelligence.pipeline import IntentClassifier

@app.post("/api/v1/process")
async def process_message(request: ProcessRequest):
    orchestrator = OrchestratorAgent()
    result = await orchestrator.process_message(request.message)
    return {"result": result}

# api/routes/analytics.py  
from core.user import AnalyticsCollector

@app.get("/api/v1/analytics/{session_id}")
async def get_analytics(session_id: str):
    analytics = AnalyticsCollector.load(session_id)
    return analytics.generate_summary_report()

# api/routes/logs.py
from tools.session_viewer import SessionLoader

@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str):
    loader = SessionLoader()
    data = loader.load_session(session_id)
    return data
```

Clear, organized imports make API development much easier!
