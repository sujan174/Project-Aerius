"""
Memory Agent - Handles All Memory Operations

This agent is responsible for storing and retrieving information from memory:
- User facts (email, timezone, preferences, defaults)
- Past conversation search
- Session history retrieval

Design Philosophy:
- STORE operations are fire-and-forget (no confirmation needed for speed)
- RETRIEVE operations return structured information
- All memory operations go through this single agent

Author: AI System
Version: 1.0
"""

from typing import List, Dict, Any, Optional
import json
from connectors.base_agent import BaseAgent


class MemoryAgent(BaseAgent):
    """
    Memory Agent - Unified interface for all memory operations

    Capabilities:
    - Store user facts (fire-and-forget, no confirmation)
    - Retrieve user facts
    - Search past conversations
    - Get session history

    Speed Optimization:
    - Store operations return immediately without confirmation
    - Orchestrator doesn't wait for response
    - Only retrieve operations return data
    """

    def __init__(self, unified_memory, verbose: bool = False):
        """
        Initialize Memory Agent

        Args:
            unified_memory: UnifiedMemory instance from orchestrator
            verbose: Enable detailed logging
        """
        super().__init__()
        self.unified_memory = unified_memory
        self.verbose = verbose

    async def initialize(self):
        """Initialize the memory agent"""
        # Memory is already initialized by orchestrator
        self.initialized = True

        if self.verbose:
            print("[MEMORY AGENT] Initialized")

    async def get_capabilities(self) -> List[str]:
        """Return memory agent capabilities"""
        return [
            "Store and retrieve user information (name, email, timezone, preferences)",
            "Search past conversation sessions",
            "Get conversation history and context",
            "Manage user defaults and settings"
        ]

    async def execute(self, instruction: str) -> str:
        """
        Execute memory operations based on natural language instruction

        Args:
            instruction: Natural language instruction like:
                - "Store user email: john@example.com"
                - "Get all user facts"
                - "Search for conversations about Jira tickets"
                - "What is the user's timezone?"

        Returns:
            str: Result or confirmation (for retrievals only)
        """
        if not self.initialized:
            return self._format_error(Exception("Memory agent not initialized"))

        try:
            instruction_lower = instruction.lower()

            # ===================================================================
            # STORE OPERATIONS (Fire-and-forget)
            # ===================================================================

            if "store" in instruction_lower or "save" in instruction_lower or "remember" in instruction_lower:
                return await self._handle_store(instruction)

            # ===================================================================
            # RETRIEVE OPERATIONS (Return data)
            # ===================================================================

            elif "get" in instruction_lower or "retrieve" in instruction_lower or "show" in instruction_lower:
                return await self._handle_retrieve(instruction)

            elif "search" in instruction_lower or "find" in instruction_lower:
                return await self._handle_search(instruction)

            elif "remove" in instruction_lower or "delete" in instruction_lower or "forget" in instruction_lower:
                return await self._handle_remove(instruction)

            else:
                # Default: try to interpret as a question about user facts
                return await self._handle_question(instruction)

        except Exception as e:
            return self._format_error(e)

    # =========================================================================
    # STORE OPERATIONS - Fire and Forget
    # =========================================================================

    async def _handle_store(self, instruction: str) -> str:
        """
        Handle store operations (fire-and-forget)

        Examples:
        - "Store user email: john@example.com"
        - "Save timezone: EST"
        - "Remember that user prefers concise responses"
        """
        try:
            # Parse instruction to extract key-value pairs
            # Simple parsing - look for patterns like "key: value"
            import re

            # Common patterns
            patterns = {
                'email': r'email[:\s]+([^\s,]+@[^\s,]+)',
                'user_email': r'email[:\s]+([^\s,]+@[^\s,]+)',
                'timezone': r'timezone[:\s]+([A-Z]{2,4})',
                'user_name': r'(?:name|user)[:\s]+([A-Za-z\s]+?)(?:\s*,|\s*$)',
                'communication_style': r'(?:style|communication)[:\s]+(concise|verbose|normal)',
                'default_project': r'(?:default\s*)?project[:\s]+([A-Z]+-\d+|[A-Za-z0-9_-]+)',
                'default_assignee': r'(?:default\s*)?assignee[:\s]+([A-Za-z\s]+)',
            }

            stored_anything = False

            for key, pattern in patterns.items():
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()

                    # Determine category
                    if key in ['user_name', 'user_email']:
                        category = 'identity'
                    elif key in ['timezone', 'communication_style']:
                        category = 'preference'
                    elif 'default' in key:
                        category = 'default'
                    else:
                        category = 'preference'

                    # Store directly (fire-and-forget)
                    self.unified_memory.set_core_fact(
                        key=key,
                        value=value,
                        category=category,
                        source='explicit'
                    )

                    stored_anything = True

                    if self.verbose:
                        print(f"[MEMORY AGENT] Stored {key}={value} (fire-and-forget)")

            if stored_anything:
                # Minimal confirmation - just acknowledge
                return "Stored"
            else:
                # Try generic key-value extraction
                kv_match = re.search(r'(\w+)[:\s]+(.+?)(?:\s*,|\s*$)', instruction)
                if kv_match:
                    key = kv_match.group(1).strip()
                    value = kv_match.group(2).strip()

                    self.unified_memory.set_core_fact(
                        key=key,
                        value=value,
                        category='preference',
                        source='explicit'
                    )

                    return "Stored"

                return "Could not parse store instruction. Please specify key and value."

        except Exception as e:
            if self.verbose:
                print(f"[MEMORY AGENT] Store error: {e}")
            # Even on error, return quickly (fire-and-forget)
            return "Store failed"

    # =========================================================================
    # RETRIEVE OPERATIONS
    # =========================================================================

    async def _handle_retrieve(self, instruction: str) -> str:
        """
        Handle retrieve operations

        Examples:
        - "Get all user facts"
        - "Show user information"
        - "Retrieve stored preferences"
        """
        try:
            # Get all core facts
            if not self.unified_memory.core_facts:
                return "No user facts stored yet."

            lines = ["User Facts:\n"]

            # Group by category
            by_category = {}
            for key, fact in self.unified_memory.core_facts.items():
                category = fact.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(f"  - {key}: {fact.value}")

            # Format output
            category_order = ["identity", "preference", "default", "context"]
            for category in category_order:
                if category in by_category:
                    lines.append(f"\n{category.title()}:")
                    lines.extend(by_category[category])

            return "\n".join(lines)

        except Exception as e:
            return self._format_error(e)

    async def _handle_search(self, instruction: str) -> str:
        """
        Handle conversation search operations

        Examples:
        - "Search for conversations about Jira"
        - "Find past discussions about tickets"
        """
        try:
            # Extract search query from instruction
            import re

            # Look for "about X", "for X", etc.
            patterns = [
                r'about\s+(.+?)(?:\s*$|,)',
                r'for\s+(.+?)(?:\s*$|,)',
                r'search\s+(.+?)(?:\s*$|,)',
            ]

            query = None
            for pattern in patterns:
                match = re.search(pattern, instruction, re.IGNORECASE)
                if match:
                    query = match.group(1).strip()
                    break

            if not query:
                query = instruction  # Use full instruction as query

            # Search episodic memory if available
            if hasattr(self.unified_memory, 'episodic_memory') and self.unified_memory.episodic_memory:
                episodes = await self.unified_memory.episodic_memory.retrieve_relevant(
                    query=query,
                    n_results=5,
                    min_similarity=0.7
                )

                if not episodes:
                    return f"No past conversations found about '{query}'"

                lines = [f"Found {len(episodes)} relevant conversation(s):\n"]
                for i, ep in enumerate(episodes, 1):
                    metadata = ep['metadata']
                    date = metadata.get('date', 'Unknown')
                    similarity = ep['similarity']

                    lines.append(f"{i}. [{date}] (similarity: {similarity})")
                    lines.append(f"   {ep['summary']}")

                    if metadata.get('agents'):
                        lines.append(f"   Agents: {metadata['agents']}")
                    lines.append("")

                return "\n".join(lines)

            # Fallback: search session store
            if hasattr(self.unified_memory, 'session_store') and self.unified_memory.session_store:
                context = self.unified_memory.session_store.get_context_for_query(query)
                return context

            return "Memory search not available"

        except Exception as e:
            return self._format_error(e)

    async def _handle_remove(self, instruction: str) -> str:
        """
        Handle remove operations

        Examples:
        - "Remove user email"
        - "Forget my timezone"
        """
        try:
            import re

            # Extract key to remove
            # Look for common fact keys
            common_keys = [
                'user_name', 'user_email', 'email', 'name',
                'timezone', 'communication_style',
                'default_project', 'default_assignee'
            ]

            for key in common_keys:
                if key in instruction.lower():
                    # Try exact match first
                    if self.unified_memory.remove_core_fact(key):
                        return f"Removed {key}"
                    # Try without "user_" prefix
                    if key.startswith('user_'):
                        short_key = key[5:]
                        if self.unified_memory.remove_core_fact(short_key):
                            return f"Removed {short_key}"

            # Try to extract key from instruction
            match = re.search(r'(?:remove|forget|delete)\s+(?:user\s+)?(\w+)', instruction, re.IGNORECASE)
            if match:
                key = match.group(1).strip()
                if self.unified_memory.remove_core_fact(key):
                    return f"Removed {key}"
                elif self.unified_memory.remove_core_fact(f"user_{key}"):
                    return f"Removed user_{key}"

            return "Could not find fact to remove. Available facts: " + \
                   ", ".join(self.unified_memory.core_facts.keys())

        except Exception as e:
            return self._format_error(e)

    async def _handle_question(self, instruction: str) -> str:
        """
        Handle questions about user facts

        Examples:
        - "What is the user's timezone?"
        - "What's my email?"
        """
        try:
            instruction_lower = instruction.lower()

            # Map question keywords to fact keys
            keyword_map = {
                'email': 'user_email',
                'timezone': 'timezone',
                'name': 'user_name',
                'style': 'communication_style',
                'project': 'default_project',
                'assignee': 'default_assignee',
            }

            for keyword, fact_key in keyword_map.items():
                if keyword in instruction_lower:
                    value = self.unified_memory.get_core_fact(fact_key)
                    if value:
                        return f"{fact_key}: {value}"
                    else:
                        return f"{fact_key} not set"

            # Fallback: show all facts
            return await self._handle_retrieve("get all")

        except Exception as e:
            return self._format_error(e)

    # =========================================================================
    # DIRECT API METHODS (For programmatic use)
    # =========================================================================

    def store_fact_sync(self, key: str, value: str, category: str = "preference") -> None:
        """
        Direct synchronous store (fire-and-forget)

        Use this for programmatic stores that don't need confirmation.
        """
        try:
            self.unified_memory.set_core_fact(
                key=key,
                value=value,
                category=category,
                source='explicit'
            )
        except Exception as e:
            if self.verbose:
                print(f"[MEMORY AGENT] Direct store error: {e}")

    def get_fact_sync(self, key: str) -> Optional[str]:
        """Direct synchronous retrieve"""
        return self.unified_memory.get_core_fact(key)

    def get_all_facts_sync(self) -> Dict[str, str]:
        """Get all facts as a simple dict"""
        return {
            key: fact.value
            for key, fact in self.unified_memory.core_facts.items()
        }
