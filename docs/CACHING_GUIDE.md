# Advanced Caching System Guide

## Overview

Project Aerius now includes a comprehensive caching system that dramatically reduces latency and API costs through four key features:

1. **Semantic Deduplication** - Similar queries hit the same cache ("show my tickets" = "list my tasks")
2. **Persistent Cache** - Cache survives restarts, stored in `.cache/` directory
3. **Cache Warming** - Preloads common patterns on startup
4. **API Response Caching** - Caches Jira, GitHub, Slack API responses with smart TTLs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HybridCache                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Exact Match  │  │  Semantic    │  │  Persistent  │         │
│  │   Cache      │  │   Cache      │  │    Cache     │         │
│  │  (In-Memory) │  │(TF-IDF Based)│  │  (On Disk)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          API Response Cache (Per-Service TTLs)           │  │
│  │  • Jira: 3-30min  • GitHub: 2-30min  • Slack: 1min-1hr │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

### 1. Semantic Deduplication

Similar queries are matched using TF-IDF embeddings with cosine similarity:

```python
Query 1: "show my jira tickets"
Query 2: "list my tasks"
Similarity: 0.87 (> threshold 0.85) → Cache Hit!
```

**Benefits:**
- Reduces duplicate LLM calls for similar intents
- Works across different phrasings
- No configuration needed

### 2. Persistent Cache

Cache is automatically saved to disk every 5 minutes:

```
.cache/
├── aerius_cache.json         # Main cache data
├── common_patterns.json       # Cache warming patterns
```

**Features:**
- Survives application restarts
- Automatic periodic saves
- JSON format for easy inspection
- Atomic writes (no corruption)

### 3. Cache Warming

Common query patterns are preloaded on startup:

**Default Patterns:**
- Jira: "show my tickets", "list tasks", "get issues"
- GitHub: "show PRs", "list repos", "open issues"
- Slack: "recent messages", "list channels"

**Custom Patterns:**
Edit `.cache/common_patterns.json`:
```json
{
  "jira_patterns": [
    "show my sprint tasks",
    "list blockers",
    "high priority bugs"
  ],
  "github_patterns": [
    "my review requests",
    "failed CI builds"
  ]
}
```

### 4. API Response Caching

Caches actual API responses with intelligent per-endpoint TTLs:

| Service | Endpoint | TTL | Reason |
|---------|----------|-----|--------|
| **Jira** | list_issues | 3 min | Frequently changing |
| | get_issue | 5 min | Moderate updates |
| | get_user | 1 hour | Rarely changes |
| **GitHub** | list_prs | 2 min | Very active |
| | get_repo | 30 min | Stable data |
| **Slack** | list_messages | 1 min | Real-time |
| | get_user | 1 hour | Static info |

## Usage Guide

### For Agent Developers

To add caching to your custom agent:

#### Option 1: Use the Mixin (Recommended)

```python
from connectors.api_cache_mixin import APICacheMixin, cache_api_call
from connectors.base_agent import BaseAgent

class MyAgent(APICacheMixin, BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service_name = "myservice"  # For cache namespacing

    # Read operations - cache automatically
    @cache_api_call(endpoint="list_items", ttl=300)
    async def list_items(self, filter: str):
        # Your API call logic
        result = await self.api_client.list(filter=filter)
        return result

    # Write operations - invalidate cache after
    @cache_api_call(endpoint="create_item", invalidate_on_write=True)
    async def create_item(self, data: Dict):
        result = await self.api_client.create(data)
        return result
```

#### Option 2: Manual Caching

```python
async def get_data(self, project_id: str):
    if not self.api_cache:
        # No cache available - call API directly
        return await self._fetch_from_api(project_id)

    # Try cache first
    cached = self.api_cache.get(
        service="myservice",
        endpoint="get_data",
        params={"project_id": project_id}
    )

    if cached:
        return cached

    # Cache miss - fetch and store
    result = await self._fetch_from_api(project_id)

    self.api_cache.set(
        service="myservice",
        endpoint="get_data",
        params={"project_id": project_id},
        response=result
    )

    return result
```

### Cache Invalidation

Invalidate cache after write operations:

```python
# After creating/updating/deleting
self.invalidate_api_cache()  # Invalidate all endpoints

# Or specific endpoint only
self.invalidate_api_cache(endpoint="list_items")
```

### For Application Users

The cache works automatically - no configuration needed!

**View Cache Stats:**
```python
# In orchestrator
stats = orchestrator.hybrid_cache.get_stats()
print(stats)
```

**Output:**
```json
{
  "exact_cache": {
    "size": 45,
    "hits": 234,
    "misses": 67,
    "hit_rate": 0.78
  },
  "semantic_cache": {
    "entries": 23,
    "semantic_hits": 45,
    "semantic_hit_rate": 0.67
  },
  "persistent_cache": {
    "size": 89,
    "last_save": "2025-11-15T10:30:00",
    "dirty": false
  },
  "api_cache": {
    "size": 156,
    "hits": 891,
    "misses": 234,
    "hit_rate": 0.79,
    "by_service": {
      "jira": 78,
      "github": 45,
      "slack": 33
    }
  }
}
```

## Performance Impact

### Before Caching

```
User: "Show my Jira tickets"
→ LLM Classification: 200ms
→ API Call: 500ms
→ Total: 700ms

User: "List my tasks" (similar query)
→ LLM Classification: 200ms
→ API Call: 500ms
→ Total: 700ms
```

### After Caching

```
User: "Show my Jira tickets"
→ LLM Classification: 200ms
→ API Call: 500ms
→ Cache Store
→ Total: 700ms (first time)

User: "List my tasks" (similar query)
→ Semantic Cache Hit: <1ms
→ Total: <1ms (700x faster!)
```

## Cost Savings

### API Costs

With 70% cache hit rate on Jira API calls:
- **Before**: 1000 requests/day × $0.001 = $1.00/day
- **After**: 300 requests/day × $0.001 = $0.30/day
- **Savings**: $0.70/day = $21/month = $252/year

### LLM Costs

With 65% semantic cache hit rate:
- **Before**: 1000 classifications × $0.0001 = $0.10/day
- **After**: 350 classifications × $0.0001 = $0.035/day
- **Savings**: $0.065/day = $1.95/month = $23/year

### Total Savings
- ~$275/year per active user
- Scales with usage volume

## Advanced Configuration

### Custom Embedding Model

Replace TF-IDF with better embeddings:

```python
from openai import OpenAI

def openai_embeddings(text: str) -> List[float]:
    client = OpenAI()
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Initialize with custom embeddings
hybrid_cache = HybridCache(
    embedding_model=openai_embeddings,
    enable_semantic=True
)
```

### Adjust TTLs

Customize cache lifetimes:

```python
# In advanced_cache.py
api_cache = APIResponseCache(
    default_ttl=600,  # 10 minutes default
    service_ttls={
        'jira': {
            'list_issues': 60,    # 1 minute (more aggressive)
            'get_issue': 120,     # 2 minutes
        }
    }
)
```

### Disable Features

```python
# Disable semantic caching
hybrid_cache = HybridCache(
    enable_semantic=False,  # Skip semantic matching
    enable_persistent=True,
    enable_api_cache=True
)

# Disable persistence (memory only)
hybrid_cache = HybridCache(
    enable_semantic=True,
    enable_persistent=False,  # Don't save to disk
    enable_api_cache=True
)
```

## Monitoring & Debugging

### Enable Verbose Logging

```python
orchestrator = OrchestratorAgent(
    verbose=True  # Shows cache hits/misses
)
```

**Output:**
```
[CACHE] Hit: intent:7a3f2e1... (accessed 3 times)
[SEMANTIC CACHE] Found similar query: 'show my tickets' (similarity: 0.87)
[API CACHE] Hit: jira.list_issues (age: 45.2s)
[PERSISTENT CACHE] Saved 156 entries to .cache/aerius_cache.json
```

### Inspect Cache Files

```bash
# View persistent cache
cat .cache/aerius_cache.json | jq .

# View cache patterns
cat .cache/common_patterns.json | jq .

# Check cache size
du -sh .cache/
```

### Clear Cache

```python
# Clear all caches
orchestrator.hybrid_cache.exact_cache.clear()
orchestrator.hybrid_cache.semantic_cache.cleanup_expired()
orchestrator.hybrid_cache.persistent_cache.cleanup_expired()

# Or just delete cache directory
rm -rf .cache/
```

## Best Practices

### 1. Cache Read Operations, Invalidate on Write

```python
# ✓ Good: Cache reads
@cache_api_call(endpoint="get_ticket", ttl=300)
async def get_ticket(self, ticket_id):
    ...

# ✓ Good: Invalidate after writes
@cache_api_call(endpoint="update_ticket", invalidate_on_write=True)
async def update_ticket(self, ticket_id, data):
    ...
```

### 2. Use Appropriate TTLs

- **Real-time data** (messages): 1-2 minutes
- **Frequently changing** (tickets, PRs): 3-5 minutes
- **Stable data** (users, repos): 30-60 minutes
- **Static data** (config): Hours or no expiry

### 3. Warm Critical Paths

Add your most common queries to `common_patterns.json` for instant startup performance.

### 4. Monitor Hit Rates

Aim for:
- **Exact cache**: >70% hit rate
- **Semantic cache**: >60% hit rate
- **API cache**: >75% hit rate

If hit rates are low, investigate TTLs and query patterns.

## Troubleshooting

### Cache Not Persisting

**Issue**: Cache clears on restart

**Solution**: Check `.cache/` directory permissions:
```bash
ls -la .cache/
chmod 755 .cache/
```

### Low Semantic Hit Rate

**Issue**: Similar queries not matching

**Solution**: Adjust similarity threshold:
```python
semantic_cache = SemanticCache(
    similarity_threshold=0.80  # Lower = more matches
)
```

### High Memory Usage

**Issue**: Cache growing too large

**Solution**: Reduce max size:
```python
hybrid_cache = HybridCache(
    enable_persistent=True  # Move to disk
)

# Or reduce cache size
exact_cache = IntelligentCache(
    max_size=500  # Down from 1000
)
```

### Stale Data

**Issue**: Seeing old API responses

**Solution**: Reduce TTLs or invalidate manually:
```python
# Invalidate specific service
orchestrator.hybrid_cache.invalidate_api_service('jira')

# Or reduce TTL
api_cache.service_ttls['jira']['list_issues'] = 60  # 1 minute
```

## Implementation Details

### File Structure

```
core/
├── advanced_cache.py          # Main caching implementation
│   ├── SemanticCache         # Similarity-based matching
│   ├── PersistentCache       # Disk storage
│   ├── CacheWarmer           # Preloading
│   ├── APIResponseCache      # API caching
│   └── HybridCache           # Unified interface
└── simple_embeddings.py       # TF-IDF embeddings

connectors/
└── api_cache_mixin.py         # Agent integration
    ├── APICacheMixin         # Mixin class
    └── cache_api_call        # Decorator
```

### Dependencies

No external dependencies required! Uses only Python stdlib:
- `json` - Persistence
- `hashlib` - Cache keys
- `threading` - Thread safety
- `numpy` - Vector operations (for embeddings)

## Migration Guide

### Existing Agents

Your agents work as-is! Caching is opt-in:

```python
# Before (still works)
class MyAgent(BaseAgent):
    async def get_data(self):
        return await self.api_call()

# After (with caching)
class MyAgent(APICacheMixin, BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service_name = "myservice"

    @cache_api_call(endpoint="get_data", ttl=300)
    async def get_data(self):
        return await self.api_call()
```

### Gradual Rollout

1. **Phase 1**: Enable exact + persistent cache (low risk)
2. **Phase 2**: Add semantic caching (monitor quality)
3. **Phase 3**: Enable API response caching (test per-service)
4. **Phase 4**: Implement cache warming

## Future Enhancements

Planned improvements:
- [ ] Redis backend for distributed caching
- [ ] Cache analytics dashboard
- [ ] Automatic pattern learning
- [ ] Query rewriting suggestions
- [ ] Cache preheating based on schedule
- [ ] Multi-tenant cache isolation

## Support

Questions or issues? Check:
- Verbose logging: `verbose=True`
- Cache stats: `get_stats()`
- Cache files: `.cache/` directory

For bugs, open an issue with cache stats and logs.
