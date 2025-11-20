# Project Friday - AI Workspace Orchestrator - Complete Codebase Analysis

## Executive Summary

Project Friday is a sophisticated **AI Workspace Orchestrator** that serves as an intelligent multi-agent coordination system. It acts as a central hub that understands user intents, decomposes complex tasks, and orchestrates specialized AI agents to interact with various workspace tools and services (Slack, Jira, GitHub, Notion, Google Calendar, etc.).

**Total Project Size**: 2.2 MB  
**Total Python Files**: 52  
**Total Lines of Code**: ~27,000 LOC  
**Primary Language**: Python 3  
**Current Branch**: claude/code-review-refactor-01WueirqAPGMBzcjmGctpMKQ  

---

## 1. OVERALL ARCHITECTURE

### Architecture Pattern: Hub-and-Spoke Multi-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│           (Enhanced Terminal UI with Rich)               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 ORCHESTRATOR AGENT                       │
│  - Hybrid Intelligence System v5.0 (Fast Filter + LLM)   │
│  - Circuit Breaker Pattern                              │
│  - Smart Retry Management                               │
│  - Undo System                                          │
│  - User Preference Learning                             │
│  - Analytics & Observability                            │
└────┬────────────┬──────────────┬──────────────┬──────────┘
     │            │              │              │
  ┌──▼──┐    ┌────▼───┐    ┌────▼───┐    ┌────▼───┐
  │Slack│    │ Jira   │    │ GitHub  │    │ Notion │
  │Agent│    │ Agent  │    │ Agent   │    │ Agent  │
  └─────┘    └────────┘    └────────┘    └────────┘
     │            │              │              │
  ┌──▼──┐    ┌────▼───┐    ┌────▼───┐    ┌────▼────┐
  │Google│   │Code    │    │Browser  │    │Scraper  │
  │Cal   │   │Review  │    │Agent    │    │Agent    │
  └──────┘   └────────┘    └────────┘    └─────────┘
```

### Core Design Philosophy

- **Abstraction-First**: Base classes (BaseAgent, BaseLLM) enforce consistent interfaces
- **Composable Intelligence**: Intelligence components can be mixed/matched/enhanced
- **Async-First**: All I/O operations designed for async/concurrent execution
- **Observability-Built-in**: Tracing, metrics, logging integrated throughout
- **Graceful Degradation**: System continues working even if components fail
- **Provider-Agnostic**: Can swap LLM providers (Gemini → OpenAI, etc.)

---

## 2. TECHNOLOGIES & FRAMEWORKS

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Async Runtime** | asyncio |
| **LLM Provider** | Google Gemini Flash (google.generativeai) |
| **LLM Abstraction** | Custom BaseLLM interface |
| **Protocol** | Model Context Protocol (MCP) |
| **API Interaction** | MCP ClientSession, StdioServerParameters |
| **Terminal UI** | Rich library (tables, progress bars, colors) |
| **Config Management** | python-dotenv (.env files) |
| **Data Structures** | dataclasses, enums, custom type hints |
| **Persistence** | JSON files (workspace_knowledge.json, user preferences) |

### Key Dependencies (Inferred from Code)

- `google-generativeai` - Gemini API client
- `python-dotenv` - Environment configuration
- `rich` - Advanced terminal UI components
- `mcp` (Model Context Protocol) - Agent communication protocol
- No external database (uses JSON for persistence)
- No web framework (CLI-only interface)

---

## 3. PROJECT STRUCTURE IN DETAIL

### Root Files (3 files)

```
/home/user/Project-Friday/
├── main.py                    # Entry point - async main() function
├── config.py                  # Global Config class - centralized settings
├── orchestrator.py            # Main OrchestratorAgent class (1806 lines)
├── .env.example               # Configuration template with 140+ settings
└── .project-structure         # Project documentation
```

#### config.py
- **Purpose**: Centralized configuration management
- **Key Features**:
  - Reads from environment variables
  - Timeout configurations (agent, enrichment, LLM)
  - Retry strategy parameters
  - Input validation limits
  - Per-module log level overrides
  - Security and enrichment settings

#### orchestrator.py
- **Purpose**: Main orchestration engine (1806 lines)
- **Key Classes**:
  - `OrchestratorAgent`: Main orchestration class
- **Key Methods**:
  - `__init__()`: Initialize with agent discovery
  - `discover_and_load_agents()`: Dynamically load agents from connectors dir
  - `process_message()`: Main message processing pipeline
  - `call_sub_agent()`: Execute specialized agents
  - `_process_with_intelligence()`: Run hybrid intelligence pipeline

### Directory 1: /core (17 files, ~6,982 LOC)

**Purpose**: Core infrastructure, utilities, and cross-cutting concerns

#### Logging System (3 files)
- **logging_config.py** (599 LOC)
  - Enhanced production logging with structured output
  - Multiple output formats (console, file, JSON)
  - Context tracking (session, operation, agent)
  - Distributed tracing integration
  - Log rotation and archiving
  
- **simple_session_logger.py** (508 LOC)
  - Unified session logging system
  - Creates 2 files per chat session:
    1. `session_{id}_messages.jsonl` - Message flow
    2. `session_{id}_intelligence.jsonl` - Intelligence status
  
- **intelligence_logger.py** (731 LOC)
  - Tracks intelligence system metrics
  - Confidence tracking
  - Error metrics aggregation

#### Error Handling & Validation (3 files)
- **error_handler.py** (497 LOC)
  - ErrorClassifier with error pattern matching
  - ErrorCategories: TRANSIENT, RATE_LIMIT, CAPABILITY, PERMISSION, VALIDATION, UNKNOWN
  - DuplicateOperationDetector to prevent idempotency issues
  
- **error_messaging.py** (380 LOC)
  - ErrorMessageEnhancer for user-friendly error messages
  - Context-aware error explanations
  
- **input_validator.py** (5031 bytes)
  - Input sanitization and validation
  - Size limits, regex pattern limits
  - Security checks for injection attacks

#### Advanced Features (5 files)
- **analytics.py** (461 LOC)
  - AnalyticsCollector for metrics tracking
  - Operation success rates, latency tracking
  - Cost analytics
  
- **retry_manager.py** (332 LOC)
  - Smart retry logic with exponential backoff
  - Configurable backoff multiplier and initial delay
  - jitter support to prevent thundering herd
  
- **circuit_breaker.py** (377 LOC)
  - Circuit breaker pattern implementation
  - Prevents cascading failures
  - States: CLOSED (working), OPEN (failing), HALF_OPEN (testing)
  
- **undo_manager.py** (398 LOC)
  - Reversible operations support
  - UndoableOperationType enum
  - Custom undo handlers per operation type
  
- **user_preferences.py** (364 LOC)
  - Learn and remember user preferences
  - Persistent storage in JSON
  - Adaptive system behavior

#### Observability (4 files)
- **observability.py** (383 LOC)
  - Tracing initialization and management
  - Span creation and management
  - SpanKind enum (CLIENT, SERVER, INTERNAL, etc.)
  
- **distributed_tracing.py** (591 LOC)
  - Distributed tracing context
  - TraceContext for cross-service tracing
  - OpenTelemetry-compatible patterns
  
- **metrics_aggregator.py** (517 LOC)
  - Aggregate metrics across operations
  - Performance tracking
  
- **orchestration_logger.py** (641 LOC)
  - Specialized logging for orchestration events

#### Utilities
- **logger.py** (1724 bytes)
  - Backward compatibility wrapper
  - get_logger() convenience function

---

### Directory 2: /connectors (12 files, ~11,981 LOC)

**Purpose**: Integration agents for external services

#### Base Classes (3 files)
- **base_agent.py** (572 LOC)
  - Abstract BaseAgent class - enforces consistent interface
  - Abstract methods: initialize(), get_capabilities(), execute(), cleanup()
  - Error classes: AgentError, AgentNotInitializedError, AgentConnectionError, etc.
  - safe_extract_response_text() helper for Gemini responses
  
- **base_connector.py** (168 LOC)
  - BaseConnector ABC with tool definition support
  - ToolDefinition dataclass
  - ConnectorMetadata dataclass
  
- **agent_intelligence.py** (561 LOC)
  - ConversationMemory: Remember recent operations and resolve references
  - WorkspaceKnowledge: Learn and persist workspace-specific knowledge
  - SharedContext: Enable cross-agent coordination
  - ProactiveAssistant: Suggest next steps and validate operations

#### Specialized Agents (9 files)

1. **slack_agent.py** (1763 LOC)
   - MCP-based Slack integration
   - Operations: send messages, search, channel management, user management
   - Features: metadata caching, smart message formatting, retry logic
   - Uses: mcp, asyncio, Gemini for natural language understanding

2. **jira_agent.py** (1708 LOC)
   - MCP-based Jira integration
   - Operations: create issues, update, transition, search, link issues
   - Features: issue resolution, batch operations, custom field handling
   - Undo support for create, update, delete, transition operations

3. **github_agent.py** (1673 LOC)
   - MCP-based GitHub integration
   - Operations: create PRs, issues, comments, manage repos
   - Features: branch creation, file operations, merge management
   - Undo support for close PR/issue operations

4. **notion_agent.py** (1570 LOC)
   - MCP-based Notion integration
   - Operations: create/update databases, pages, queries
   - Features: rich property support, database queries, relation handling
   - Undo support for delete page operations

5. **google_calendar_agent.py** (1463 LOC)
   - OAuth2-based Google Calendar integration
   - Operations: create events, search, list calendars, update events
   - Features: attendee management, reminder configuration
   - Manual auth flow support for headless environments

6. **scraper_agent.py** (809 LOC)
   - Web scraping capabilities
   - Operations: fetch page content, extract data, handle redirects
   - Features: charset detection, rate limiting, retry logic

7. **browser_agent.py** (791 LOC)
   - Browser automation capabilities
   - Operations: navigation, form filling, element interaction
   - Features: headless mode support, screenshot capability

8. **code_reviewer_agent.py** (716 LOC)
   - AI-powered code review
   - Operations: review code, suggest improvements, check quality
   - Uses Gemini for intelligent code analysis

#### Configuration
- **mcp_config.py** (187 LOC)
  - MCPTimeouts: Connection, operation timeouts for all agents
  - MCPRetryConfig: Retry strategy for MCP operations
  - MCPRetryableErrors: Patterns for automatic retry (network, SSE, HTTP, rate limit, server)
  - MCPErrorMessages: Standardized error message formatting
  - MCPVerboseLogger: Consistent logging across MCP agents

---

### Directory 3: /intelligence (12 files, ~6,500 LOC)

**Purpose**: AI intelligence components for understanding and decision-making

#### Base Types & Infrastructure (3 files)
- **base_types.py** (706 LOC)
  - IntentType enum: CREATE, READ, UPDATE, DELETE, ANALYZE, COORDINATE, WORKFLOW, SEARCH
  - Intent dataclass: type, confidence, entities, implicit_requirements
  - EntityType enum: PROJECT, PERSON, TEAM, RESOURCE, DATE, PRIORITY, STATUS, LABEL, ISSUE, PR, CHANNEL, REPOSITORY, FILE, CODE
  - Entity dataclass: type, value, confidence, context
  - Task, ExecutionPlan, DependencyGraph for task decomposition
  - Confidence and ConfidenceLevel for decision confidence tracking

- **cache_layer.py** (353 LOC)
  - Global caching for intelligence results
  - CacheEntry dataclass with TTL
  - CacheKeyBuilder for consistent cache key generation
  - get_global_cache() singleton accessor

- **hybrid_system.py** (364 LOC)
  - Hybrid Intelligence System v5.0 - Two-tier architecture
  - Tier 1: FastKeywordFilter (~10ms, free, 35-40% coverage)
  - Tier 2: LLMIntentClassifier (~200ms, paid, 60-65% coverage)
  - HybridIntelligenceResult dataclass
  - Coordinates fast vs. LLM paths based on confidence

#### Intelligence Components (5 files)
- **intent_classifier.py** (686 LOC)
  - IntentClassifier: Understand user intent
  - Bayesian confidence estimation
  - Intent patterns and keywords
  - Multi-intent detection

- **entity_extractor.py** (756 LOC)
  - EntityExtractor: Extract structured information
  - Entity type classification
  - Relationship extraction
  - Context-aware extraction

- **task_decomposer.py** (481 LOC)
  - TaskDecomposer: Break complex tasks into steps
  - Dependency graph generation
  - Execution plan creation
  - Sequential vs. parallel task execution

- **confidence_scorer.py** (728 LOC)
  - ConfidenceScorer: Confidence estimation using Bayesian methods
  - Risk assessment
  - Uncertainty quantification
  - Decision confidence metrics

- **context_manager.py** (418 LOC)
  - ConversationContextManager: Maintain conversation state
  - Context sliding window
  - Relevant context extraction
  - Multi-turn conversation support

#### Specialized Classifiers (2 files)
- **fast_filter.py** (357 LOC)
  - FastKeywordFilter: Quick pattern-based classification
  - Regex-based intent matching
  - No API calls (free)
  - Fallback to LLM if low confidence

- **llm_classifier.py** (360 LOC)
  - LLMIntentClassifier: Use LLM for intent classification
  - Gemini-based semantic understanding
  - Function calling support
  - Confidence tracking

#### Documentation
- **REFACTORING_REPORT.md** (36KB)
  - Complete v2.0 → v3.0 refactoring documentation
  - Performance improvements (40-60%)
  - Architecture overview
  - Usage examples
  - Migration guide

---

### Directory 4: /llms (2 files)

**Purpose**: LLM abstraction layer (provider-agnostic)

- **base_llm.py** (314 LOC)
  - Abstract BaseLLM class - enforces consistent LLM interface
  - LLMConfig dataclass: model_name, temperature, max_tokens, etc.
  - ChatMessage dataclass: role (user/assistant/system), content
  - FunctionCall dataclass: name, arguments
  - LLMResponse dataclass: text, function_calls, finish_reason, metadata
  - Abstract methods: generate(), generate_with_tools(), stream(), count_tokens()

- **gemini_flash.py** (401 LOC)
  - GeminiFlash: Concrete implementation using Google Gemini
  - Implements all BaseLLM abstract methods
  - Function calling / tool use support
  - Token counting via genai API
  - Model: "gemini-2.0-flash" (fast, affordable)
  - Error handling for API failures

---

### Directory 5: /ui (2 files)

**Purpose**: User interface components

- **enhanced_terminal_ui.py** (412 LOC)
  - EnhancedTerminalUI: Rich-based beautiful terminal interface
  - Components: Header, Agent summary table, Response formatting
  - Features: Colored output, progress indicators, formatted tables
  - Methods: print_header(), print_tool_call(), print_response(), print_error()

- **terminal_ui.py** (305 LOC)
  - TerminalUI: Original simple terminal interface (fallback)
  - Colors class: Color codes (GREEN, CYAN, MAGENTA, etc.)
  - Icons class: Terminal icons for visual feedback

---

### Directory 6: /tools (1 file)

- **session_viewer.py** (786 LOC)
  - SessionViewer: Tool to analyze and visualize session logs
  - Reads session_{id}_messages.jsonl and session_{id}_intelligence.jsonl
  - Displays message flow, intelligence metrics, performance stats
  - Command: `python tools/session_viewer.py <session_id>`

---

### Directory 7: /scripts (1 file)

- **migrate_logging.py** (395 LOC)
  - Logging system migration utility
  - Helps migrate from old logging format to new unified session logging
  - One-time utility script

---

### Directory 8: /docs (5 markdown files)

- **LOGGING_SYSTEM.md**: Unified session logging specification
- **LOGGING_GUIDE.md**: How to use the logging system
- **SESSION_VIEWER_GUIDE.md**: How to use session viewer tool
- **LOGGING_SYSTEM_README.md**: Logging system overview
- **ORCHESTRATOR_LOGGING_INTEGRATION.md**: Logging integration details

---

### Directory 9: /orchestration (Empty)

- **__init__.py**: Package marker (currently empty)
- Purpose reserved for future orchestration patterns

---

## 4. ENTRY POINTS & MAIN FILES

### Primary Entry Point: main.py

```python
async def main():
    # Parse command-line arguments
    # Initialize EnhancedTerminalUI or fallback to simple UI
    # Create OrchestratorAgent instance
    # Discover and load agents
    # Run interactive session
    # Handle cleanup

if __name__ == "__main__":
    asyncio.run(main())
```

**Usage**:
```bash
python main.py              # Start with enhanced UI
python main.py --verbose    # Show debug information
python main.py --simple     # Use simple UI (fallback)
```

### Secondary Entry Points:

1. **Orchestrator Agent**
   ```python
   orchestrator = OrchestratorAgent(
       connectors_dir="connectors",
       verbose=False
   )
   response = await orchestrator.process_message(user_message)
   ```

2. **Session Viewer**
   ```bash
   python tools/session_viewer.py <session_id>
   ```

---

## 5. CONFIGURATION FILES

### .env.example (140+ configuration options)

Configuration is organized into 10 sections:

1. **API Keys**: GOOGLE_API_KEY, NOTION_TOKEN, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
2. **Confirmation Settings**: CONFIRM_SLACK_MESSAGES, CONFIRM_JIRA_OPERATIONS, CONFIRM_DELETES, etc.
3. **Agent Operation Settings**: Timeouts for agents, enrichment, LLM
4. **Batch Processing**: Batch timeout, max batch size, max pending actions
5. **Retry Configuration**: Max retries, backoff multiplier, initial delay
6. **Logging Configuration**: Log level, log directory, file/JSON/console logging, colored logs
7. **Per-Module Log Levels**: LOG_LEVEL_ORCHESTRATOR, LOG_LEVEL_SLACK, etc.
8. **Security Settings**: Input sanitization, max instruction length, max parameter length
9. **Enrichment Settings**: Require enrichment for high-risk ops, fail open behavior

### config.py

- Reads all environment variables with defaults
- Single Config class - no object instantiation needed
- Class attributes for all settings
- get_config() method returns all settings as dict

---

## 6. TESTING STRUCTURE

**Status**: NO AUTOMATED TESTS in the codebase

- No test directory
- No pytest.ini, conftest.py, or test_*.py files
- No unit tests or integration tests
- No CI/CD pipeline configuration

**Note**: Manual testing and session logs are the primary verification mechanism.

---

## 7. KEY ARCHITECTURAL PATTERNS

### 1. Multi-Agent Orchestration Pattern
- Central orchestrator coordinates specialized agents
- Each agent handles one domain (Slack, Jira, GitHub, etc.)
- Agents discover dynamically from connectors/ directory
- Async execution for concurrent operations

### 2. Hybrid Intelligence System v5.0
- **Two-Tier Decision Making**:
  1. **Fast Filter** (Tier 1): Keyword-based classification (~10ms, free)
  2. **LLM Classifier** (Tier 2): Semantic understanding (~200ms, paid)
- Intelligently chooses path based on confidence
- Caches results to avoid redundant processing
- Falls back to LLM for ambiguous cases

### 3. Abstract Base Classes for Extension
- **BaseAgent**: All agents inherit from this
- **BaseLLM**: All LLM providers implement this
- **BaseConnector**: Foundation for tool definitions
- Enforces consistent interface across implementations

### 4. Error Handling & Recovery
- **ErrorClassifier**: Categorize errors (transient, capability, permission, etc.)
- **RetryManager**: Smart retries with exponential backoff
- **CircuitBreaker**: Prevent cascading failures
- **DuplicateOperationDetector**: Prevent idempotency issues

### 5. Observability & Monitoring
- **SimpleSessionLogger**: JSONL-based session logging
- **DistributedTracing**: OpenTelemetry-compatible tracing
- **AnalyticsCollector**: Metrics tracking (success rate, latency, cost)
- **Per-module log levels**: Fine-grained control

### 6. Undo System
- **UndoManager**: Reversible operations
- Register custom undo handlers per operation type
- Built-in handlers for common operations (delete, transition, etc.)

### 7. Adaptive Learning
- **UserPreferenceManager**: Learn user preferences over time
- Persistent storage in JSON
- Influences system behavior and suggestions

### 8. Provider-Agnostic Design
- LLM abstraction allows swapping providers
- MCP protocol for agent communication
- Custom base classes for extension

---

## 8. DATA FLOW & EXECUTION PATH

### Typical User Message Processing

```
1. USER INPUT
   ↓
2. INPUT VALIDATION (InputValidator)
   - Check length, pattern, sanitize
   ↓
3. HYBRID INTELLIGENCE PROCESSING
   ├─ Try Fast Filter (FastKeywordFilter)
   │  ├─ Check confidence level
   │  └─ If high confidence → return (skip LLM)
   └─ Try LLM Classifier (LLMIntentClassifier)
      ├─ Call Gemini API
      ├─ Extract intent, entities
      └─ Estimate confidence
   ↓
4. LOGGING
   - SimpleSessionLogger: Log in messages.jsonl
   - AnalyticsCollector: Track metrics
   ↓
5. ORCHESTRATION
   ├─ IntentClassifier: Determine intent (CREATE, READ, etc.)
   ├─ EntityExtractor: Extract entities (projects, people, dates)
   ├─ TaskDecomposer: Break into steps
   └─ ConfidenceScorer: Estimate confidence
   ↓
6. AGENT SELECTION & ROUTING
   - Match intent + entities to best agent(s)
   - Example: "Create Jira issue" → use JiraAgent
   ↓
7. AGENT EXECUTION (with retry/circuit breaker)
   ├─ RetryManager: Retry on transient failure
   ├─ CircuitBreaker: Stop if repeated failures
   └─ MCP-based tool execution
   ↓
8. RESPONSE FORMATTING
   - Format agent response for user
   - Extract key information
   ↓
9. LOGGING
   - Log final response
   - Store in messages.jsonl
   ↓
10. USER OUTPUT
    - Display via EnhancedTerminalUI
```

---

## 9. DIRECTORY TREE (Complete)

```
/home/user/Project-Friday/
├── main.py                                  # Entry point
├── orchestrator.py                          # Main orchestration engine (1806 LOC)
├── config.py                                # Global configuration
├── .env.example                             # Configuration template
├── .project-structure                       # Documentation
├── .gitignore                               # Git ignore rules
├── CLAUDE.md                                # Empty marker file
│
├── core/                                    # Core infrastructure (17 files)
│   ├── __init__.py
│   ├── logger.py                            # Logging wrapper
│   ├── logging_config.py                    # Enhanced logging system
│   ├── simple_session_logger.py             # Session logging
│   ├── intelligence_logger.py               # Intelligence metrics logging
│   ├── orchestration_logger.py              # Orchestration logging
│   ├── error_handler.py                     # Error classification
│   ├── error_messaging.py                   # User-friendly errors
│   ├── input_validator.py                   # Input validation
│   ├── analytics.py                         # Analytics collection
│   ├── retry_manager.py                     # Retry logic
│   ├── circuit_breaker.py                   # Circuit breaker pattern
│   ├── undo_manager.py                      # Undo system
│   ├── user_preferences.py                  # User preferences
│   ├── observability.py                     # Tracing/observability
│   ├── distributed_tracing.py               # Distributed tracing
│   └── metrics_aggregator.py                # Metrics aggregation
│
├── connectors/                              # Agent connectors (12 files)
│   ├── base_agent.py                        # Base agent class
│   ├── base_connector.py                    # Base connector class
│   ├── agent_intelligence.py                # Shared intelligence components
│   ├── slack_agent.py                       # Slack integration (1763 LOC)
│   ├── jira_agent.py                        # Jira integration (1708 LOC)
│   ├── github_agent.py                      # GitHub integration (1673 LOC)
│   ├── notion_agent.py                      # Notion integration (1570 LOC)
│   ├── google_calendar_agent.py             # Google Calendar (1463 LOC)
│   ├── scraper_agent.py                     # Web scraping (809 LOC)
│   ├── browser_agent.py                     # Browser automation (791 LOC)
│   ├── code_reviewer_agent.py               # Code review (716 LOC)
│   └── mcp_config.py                        # MCP configuration
│
├── intelligence/                            # AI intelligence (12 files)
│   ├── __init__.py
│   ├── base_types.py                        # Data structures
│   ├── hybrid_system.py                     # Hybrid intelligence v5.0
│   ├── fast_filter.py                       # Fast keyword filter
│   ├── llm_classifier.py                    # LLM classifier
│   ├── intent_classifier.py                 # Intent classification
│   ├── entity_extractor.py                  # Entity extraction
│   ├── task_decomposer.py                   # Task decomposition
│   ├── confidence_scorer.py                 # Confidence scoring
│   ├── context_manager.py                   # Context management
│   ├── cache_layer.py                       # Caching system
│   └── REFACTORING_REPORT.md                # Refactoring documentation
│
├── llms/                                    # LLM abstraction (2 files)
│   ├── base_llm.py                          # Base LLM interface
│   └── gemini_flash.py                      # Gemini implementation
│
├── ui/                                      # User interface (2 files)
│   ├── __init__.py
│   ├── enhanced_terminal_ui.py              # Rich-based UI
│   └── terminal_ui.py                       # Simple terminal UI
│
├── tools/                                   # Utility tools (1 file)
│   └── session_viewer.py                    # Session log viewer
│
├── scripts/                                 # Utility scripts (1 file)
│   ├── __init__.py
│   └── migrate_logging.py                   # Logging migration
│
├── orchestration/                           # Reserved for orchestration (empty)
│   └── __init__.py
│
├── docs/                                    # Documentation (5 files)
│   ├── LOGGING_SYSTEM.md
│   ├── LOGGING_GUIDE.md
│   ├── SESSION_VIEWER_GUIDE.md
│   ├── LOGGING_SYSTEM_README.md
│   └── ORCHESTRATOR_LOGGING_INTEGRATION.md
│
└── .git/                                    # Git repository
```

---

## 10. KEY TECHNOLOGIES SUMMARY

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Core implementation |
| **Async** | asyncio | Concurrent I/O operations |
| **LLM** | Google Gemini Flash | AI intelligence |
| **LLM Abstraction** | Custom BaseLLM | Provider independence |
| **Agent Protocol** | Model Context Protocol (MCP) | Tool calling |
| **Config** | python-dotenv | Environment configuration |
| **UI** | Rich library | Beautiful terminal interface |
| **Logging** | Python logging + custom wrappers | Structured logging |
| **Persistence** | JSON files | User preferences, workspace knowledge |
| **Data Structures** | dataclasses, enums | Type-safe data handling |
| **Testing** | None (manual only) | Manual verification via sessions |

---

## 11. IMPORTANT PATTERNS & CONVENTIONS

### Naming Conventions
- Agents: `{service_name}_agent.py` with `class Agent(BaseAgent)`
- Functions: snake_case (Python standard)
- Constants: UPPER_CASE
- Private methods: _leading_underscore()

### Configuration Pattern
- All settings in `config.py`
- Environment variable overrides with defaults
- No hardcoded values in code

### Error Handling Pattern
```python
try:
    result = await operation()
except Exception as e:
    classification = ErrorClassifier.classify(str(e), agent_name)
    if classification.is_retryable:
        # Retry with RetryManager
    else:
        # Return user-friendly error via ErrorMessageEnhancer
```

### Agent Implementation Pattern
```python
class Agent(BaseAgent):
    async def initialize(self):
        # Connect to external service
        pass
    
    def get_capabilities(self):
        # Return list of things agent can do
        return ["create_issue", "update_issue", ...]
    
    async def execute(self, instruction: str, context: Dict = None):
        # Parse instruction and execute via tools
        # Return string result
        pass
    
    async def cleanup(self):
        # Close connections
        pass
```

### Intelligence Processing Pattern
```python
# Hybrid approach
result = await hybrid_intelligence.classify_intent(user_message)

if result.path_used == 'fast':
    # Quick classification used
    pass
else:
    # LLM-based classification used
    pass

# Always log results
logger.info(f"Intent: {result.intents[0].type}, Confidence: {result.confidence}")
```

---

## 12. CONFIGURATION HIERARCHY

```
Environment Variables (highest priority)
    ↓
config.py defaults (with validation)
    ↓
.env.example reference (documentation)
```

**Key Configuration Categories**:
1. **API Credentials**: GOOGLE_API_KEY, NOTION_TOKEN, etc.
2. **Operation Control**: CONFIRM_* flags for confirmation dialogs
3. **Performance**: Timeouts, batch sizes, max actions
4. **Reliability**: Retry counts, backoff multipliers
5. **Observability**: Log levels, logging formats
6. **Security**: Sanitization, input limits

---

## 13. ASYNC EXECUTION MODEL

- All I/O operations are async (agent calls, LLM calls, logging)
- Main event loop in `main()` using `asyncio.run()`
- Concurrent agent execution possible via `asyncio.gather()`
- Timeouts enforced via `asyncio.wait_for()`
- Progress callbacks for long operations

---

## 14. SESSION & LOGGING STRUCTURE

### Two-File Session Logging

Each chat session creates:

1. **messages.jsonl** - Message flow
   ```json
   {"timestamp": "...", "type": "user_message|orchestrator_to_agent|agent_to_orchestrator|assistant_response", "from": "...", "to": "...", "content": "...", "metadata": {...}}
   ```

2. **intelligence.jsonl** - Intelligence metrics
   ```json
   {"turn_number": 1, "timestamp": "...", "intent": {...}, "entities": [...], "confidence": 0.95, "agent_selection": "jira_agent", "latency_ms": 234.5}
   ```

### Log File Organization
```
logs/
├── session_<uuid>_messages.jsonl
├── session_<uuid>_intelligence.jsonl
└── ... (one pair per session)
```

---

## 15. DEPLOYMENT & DEPENDENCIES

### Required Environment Setup

1. **Python Runtime**: 3.8+
2. **API Keys**:
   - GOOGLE_API_KEY for Gemini API
   - NOTION_TOKEN for Notion integration
   - GOOGLE_CLIENT_ID/SECRET for Google Calendar
3. **npm Packages** (for MCP servers): Via MCP configuration
4. **Virtual Environment**: Recommended

### No External Dependencies Beyond

- google-generativeai (Gemini SDK)
- python-dotenv (config)
- rich (UI)
- mcp (agent protocol)

### No External Services

- No database (JSON persistence only)
- No external caching service (in-memory caching)
- No message queue
- No container orchestration

---

## SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **Architecture** | Hub-and-spoke multi-agent orchestration |
| **Primary Language** | Python 3.8+ |
| **Total Files** | 52 Python files |
| **Total LOC** | ~27,000 lines |
| **Project Size** | 2.2 MB |
| **Entry Point** | main.py → asyncio.run(main()) |
| **Main Class** | OrchestratorAgent (1806 LOC) |
| **Agents** | 9 specialized agents (Slack, Jira, GitHub, Notion, Google Calendar, Browser, Scraper, Code Review, Base) |
| **Intelligence** | Hybrid System v5.0 (Fast Filter + LLM) |
| **LLM Provider** | Google Gemini Flash |
| **UI** | Rich-based terminal interface |
| **Logging** | Session-based JSONL logging |
| **Testing** | None (manual verification) |
| **Configuration** | 140+ options in .env.example |
| **Observability** | Built-in tracing, metrics, analytics |
| **Error Handling** | Classification-based with recovery strategies |
| **Persistence** | JSON files (no database) |

