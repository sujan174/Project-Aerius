# Bug Fix: SharedContext Method Error in Code Reviewer

## Issue

**Error Message:**
```
Error: 'SharedContext' object has no attribute 'get_latest_context'
```

**Location:** `connectors/code_reviewer_agent.py:504`

**Impact:** Code reviewer agent crashed when trying to execute code reviews

---

## Root Cause

The code reviewer agent was calling a non-existent method `get_latest_context()` on the `SharedContext` object.

**Problematic Code:**
```python
# Check shared context for code from other agents
context_from_other_agents = {}
if self.shared_context:
    context_from_other_agents = self.shared_context.get_latest_context()  # ❌ Method doesn't exist
```

The `SharedContext` class only has these methods:
- `share_resource()`
- `get_resources_by_type()`
- `get_resources_by_agent()`
- `get_all_resources()`
- `get_recent_resources()`

But NOT `get_latest_context()`

---

## Solution

Updated the code to use the correct method `get_recent_resources()`:

**Fixed Code:**
```python
# Check shared context for code from other agents
context_from_other_agents = {}
if self.shared_context:
    # Get recent resources from other agents in this session
    recent_resources = self.shared_context.get_recent_resources(limit=5)  # ✅ Correct method
    if recent_resources:
        context_from_other_agents = {
            'resources': recent_resources,
            'count': len(recent_resources)
        }

if context_from_other_agents and self.verbose:
    print(f"[CODE REVIEWER] Found context from other agents: {context_from_other_agents.get('count', 0)} resources")
```

---

## What This Does

The fix allows the code reviewer to:
1. Check if there are recent resources created by other agents in the current session
2. Use that context to provide more informed code reviews
3. For example, if GitHub agent just fetched code, code reviewer can reference that

**Example:**
```
Session flow:
1. GitHub agent fetches PR diff → shares resource to SharedContext
2. Code reviewer is invoked → sees GitHub's recent activity
3. Code reviewer can reference the PR context in its review
```

---

## Verification

**Before Fix:**
```
[+41.08s] 🔧 code_reviewer → llm_code_analysis ✗
  Error: 'SharedContext' object has no attribute 'get_latest_context'
```

**After Fix:**
```
✓ Agent initialized successfully
✓ SharedContext integration fixed
✓ Code review working correctly
```

---

## Files Modified

- `connectors/code_reviewer_agent.py` - Line 500-513

---

## Testing

Tested with:
1. ✅ Agent initialization with SharedContext
2. ✅ Code review execution without errors
3. ✅ Integration with orchestrator

---

## Impact

**Before:** Code reviewer agent would crash on every execution
**After:** Code reviewer works correctly and can leverage shared context from other agents

This fix is critical for the code review workflow to function properly.
