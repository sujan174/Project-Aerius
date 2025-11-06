# BYOT (Bring Your Own Token) Scaling Strategy

## Overview

This document outlines the architecture and implementation strategy for scaling the Notion agent to support **multi-tenant BYOT (Bring Your Own Token)** deployments. This allows each user/organization to use their own Notion integration token while sharing the same chatbot infrastructure.

## Why BYOT?

### Benefits

1. **Security & Privacy**:
   - Each user's Notion data stays isolated
   - Users control their own permissions
   - No shared credentials or cross-tenant access

2. **Scalability**:
   - No rate limit bottlenecks from shared token
   - Each tenant gets Notion's full API quota
   - Linear scaling with user growth

3. **Compliance**:
   - Meets enterprise security requirements
   - Data residency controls per tenant
   - Audit trails per organization

4. **Flexibility**:
   - Users can revoke access anytime
   - Different permission levels per user
   - Support for multiple workspaces

## Architecture

### Current (Single-Token)

```
┌─────────────┐
│  Chatbot    │
│ Orchestrator│
└──────┬──────┘
       │
       │ NOTION_TOKEN (single shared token)
       │
       ▼
┌─────────────────────┐
│  Notion MCP Server  │
│  (single instance)  │
└──────┬──────────────┘
       │
       ▼
  Notion API
  (single workspace)
```

**Limitations**:
- All users share one Notion workspace
- Single point of failure
- Rate limits affect everyone
- No user isolation

### BYOT Architecture (Recommended)

```
┌─────────────────────────────────────────┐
│          Chatbot Orchestrator           │
│                                         │
│  ┌─────────────────────────────────┐  │
│  │   Token Management Service      │  │
│  │  - User → Token mapping         │  │
│  │  - Token encryption/decryption  │  │
│  │  - Token validation & caching   │  │
│  └─────────────────────────────────┘  │
└──────┬──────────────┬──────────────────┘
       │              │
       │ user1_token  │ user2_token
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│ Notion MCP   │ │ Notion MCP   │
│ (user1)      │ │ (user2)      │
└──────┬───────┘ └──────┬───────┘
       │                │
       ▼                ▼
   Workspace 1     Workspace 2
```

**Advantages**:
- Complete user isolation
- Independent rate limits
- Fault isolation
- Scalable to thousands of users

## Implementation Guide

### Phase 1: Token Management Service

Create a service to manage user tokens securely.

#### File: `connectors/token_manager.py`

```python
"""
Token Management Service for BYOT
Handles secure storage, retrieval, and validation of user tokens
"""

import os
import json
from typing import Optional, Dict
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages user-specific API tokens with encryption

    Features:
    - Encrypted token storage
    - In-memory caching for performance
    - Token validation
    - Multi-tenant support
    """

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize token manager

        Args:
            encryption_key: Fernet key for token encryption
                          If None, uses TOKEN_ENCRYPTION_KEY env var
        """
        # Get or generate encryption key
        self.encryption_key = encryption_key or os.getenv("TOKEN_ENCRYPTION_KEY")
        if not self.encryption_key:
            # Generate a new key (save this securely!)
            self.encryption_key = Fernet.generate_key().decode()
            logger.warning(
                "No TOKEN_ENCRYPTION_KEY found. Generated new key. "
                "Save this to your .env file: TOKEN_ENCRYPTION_KEY=" + self.encryption_key
            )

        self.cipher = Fernet(self.encryption_key.encode())

        # In-memory cache: {user_id: {service: token}}
        self.token_cache: Dict[str, Dict[str, str]] = {}

        # Token storage file (encrypted)
        self.storage_file = os.getenv("TOKEN_STORAGE_FILE", "data/user_tokens.enc")

        # Load existing tokens
        self._load_tokens()

    def _load_tokens(self):
        """Load encrypted tokens from storage"""
        if not os.path.exists(self.storage_file):
            logger.info("No existing token storage found. Starting fresh.")
            return

        try:
            with open(self.storage_file, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = self.cipher.decrypt(encrypted_data)
            self.token_cache = json.loads(decrypted_data.decode())

            logger.info(f"Loaded tokens for {len(self.token_cache)} users")
        except Exception as e:
            logger.error(f"Failed to load tokens: {e}")
            self.token_cache = {}

    def _save_tokens(self):
        """Save encrypted tokens to storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)

            # Encrypt and save
            data = json.dumps(self.token_cache).encode()
            encrypted_data = self.cipher.encrypt(data)

            with open(self.storage_file, 'wb') as f:
                f.write(encrypted_data)

            logger.info(f"Saved tokens for {len(self.token_cache)} users")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def set_token(self, user_id: str, service: str, token: str):
        """
        Store a user's token for a service

        Args:
            user_id: Unique user identifier
            service: Service name (e.g., 'notion', 'github')
            token: API token to store
        """
        if user_id not in self.token_cache:
            self.token_cache[user_id] = {}

        self.token_cache[user_id][service] = token
        self._save_tokens()

        logger.info(f"Stored {service} token for user {user_id}")

    def get_token(self, user_id: str, service: str) -> Optional[str]:
        """
        Retrieve a user's token for a service

        Args:
            user_id: Unique user identifier
            service: Service name (e.g., 'notion', 'github')

        Returns:
            Token string if found, None otherwise
        """
        return self.token_cache.get(user_id, {}).get(service)

    def remove_token(self, user_id: str, service: str):
        """
        Remove a user's token for a service

        Args:
            user_id: Unique user identifier
            service: Service name (e.g., 'notion', 'github')
        """
        if user_id in self.token_cache and service in self.token_cache[user_id]:
            del self.token_cache[user_id][service]
            self._save_tokens()
            logger.info(f"Removed {service} token for user {user_id}")

    def has_token(self, user_id: str, service: str) -> bool:
        """
        Check if a user has a token for a service

        Args:
            user_id: Unique user identifier
            service: Service name

        Returns:
            True if token exists, False otherwise
        """
        return bool(self.get_token(user_id, service))

    def validate_token(self, user_id: str, service: str) -> bool:
        """
        Validate that a user's token exists and is non-empty

        Args:
            user_id: Unique user identifier
            service: Service name

        Returns:
            True if token is valid, False otherwise
        """
        token = self.get_token(user_id, service)
        return bool(token and len(token) > 0)


# Global singleton instance
_token_manager: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    """Get the global TokenManager instance"""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenManager()
    return _token_manager
```

### Phase 2: Update Notion Agent for BYOT

Modify `notion_agent.py` to accept user-specific tokens:

```python
# In notion_agent.py, update the __init__ method:

class Agent(BaseAgent):
    def __init__(
        self,
        verbose: bool = False,
        shared_context: Optional[SharedContext] = None,
        knowledge_base: Optional[WorkspaceKnowledge] = None,
        session_logger=None,
        user_id: Optional[str] = None,  # NEW: User identifier for BYOT
        user_token: Optional[str] = None,  # NEW: User-specific token
    ):
        super().__init__()

        self.logger = session_logger
        self.agent_name = "notion"
        self.user_id = user_id  # Store user context
        self.user_token = user_token  # Store user token

        # ... rest of initialization


# In the initialize() method, use user_token instead of env var:

async def initialize(self):
    """Initialize with user-specific token"""

    # Get token from user_token parameter, token manager, or env var
    if self.user_token:
        notion_token = self.user_token
    elif self.user_id:
        from connectors.token_manager import get_token_manager
        token_manager = get_token_manager()
        notion_token = token_manager.get_token(self.user_id, 'notion')
        if not notion_token:
            raise ValueError(
                f"No Notion token found for user {self.user_id}. "
                "Please register your Notion integration token."
            )
    else:
        # Fallback to environment variable (single-token mode)
        notion_token = os.getenv("NOTION_TOKEN")
        if not notion_token:
            raise ValueError("NOTION_TOKEN required")

    # Use the user-specific token
    env_vars = {**os.environ}
    env_vars["NOTION_TOKEN"] = notion_token

    # ... rest of initialization
```

### Phase 3: User Registration Flow

Create an endpoint/command for users to register their tokens:

```python
# Example user registration flow

async def register_notion_token(user_id: str, notion_token: str) -> Dict:
    """
    Register a user's Notion integration token

    Args:
        user_id: Unique user identifier (email, username, org ID, etc.)
        notion_token: Notion integration token (starts with "secret_")

    Returns:
        Dict with registration status
    """
    from connectors.token_manager import get_token_manager

    # Validate token format
    if not notion_token.startswith("secret_"):
        return {
            "success": False,
            "error": "Invalid token format. Must start with 'secret_'"
        }

    # Test token by creating a temporary agent
    try:
        agent = NotionAgent(user_token=notion_token, verbose=False)
        await agent.initialize()
        await agent.cleanup()

        # Token works! Save it
        token_manager = get_token_manager()
        token_manager.set_token(user_id, 'notion', notion_token)

        return {
            "success": True,
            "message": "Notion integration registered successfully"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Token validation failed: {str(e)}"
        }
```

### Phase 4: Orchestrator Integration

Update the orchestrator to pass user context:

```python
# In orchestrator.py

class Orchestrator:
    def __init__(self):
        # ... existing init

        # Add token manager
        from connectors.token_manager import get_token_manager
        self.token_manager = get_token_manager()

    async def _initialize_notion_agent(self, user_id: Optional[str] = None):
        """Initialize Notion agent with user context"""

        # Get user-specific token if user_id provided
        user_token = None
        if user_id:
            user_token = self.token_manager.get_token(user_id, 'notion')

        # Initialize agent with user token
        agent = NotionAgent(
            verbose=self.verbose,
            shared_context=self.shared_context,
            knowledge_base=self.knowledge,
            session_logger=self.session_logger,
            user_id=user_id,
            user_token=user_token
        )

        await agent.initialize()
        return agent
```

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Token encryption key (generate once and keep secret)
TOKEN_ENCRYPTION_KEY=your_fernet_key_here

# Token storage file path
TOKEN_STORAGE_FILE=data/user_tokens.enc

# Enable BYOT mode
ENABLE_BYOT=true

# Fallback to single token if user has none
BYOT_FALLBACK_TO_ENV=false
```

### Generating Encryption Key

```python
from cryptography.fernet import Fernet
key = Fernet.generate_key()
print(f"TOKEN_ENCRYPTION_KEY={key.decode()}")
```

Add this to your `.env` file and **keep it secret!**

## User Onboarding Flow

### Step 1: User Creates Notion Integration

Guide users through:
1. Go to https://www.notion.so/profile/integrations
2. Create "New Integration"
3. Configure permissions
4. Copy token

### Step 2: User Registers Token

Provide a command or UI:

```
User: /register-notion
Bot: Please provide your Notion integration token.
     (Visit https://www.notion.so/profile/integrations to create one)

User: secret_abc123xyz...
Bot: ✅ Validating token...
     ✅ Token validated successfully!
     ✅ Notion integration registered for your account.

     Next steps:
     1. Share your Notion pages with the integration
     2. Try: "List my Notion pages"
```

### Step 3: User Shares Pages

Users must share pages with their integration:
- Open page in Notion
- Click "···" → "Connections"
- Add their integration

## Scaling Considerations

### Performance

**Connection Pooling**:
- Maintain pool of MCP server instances
- Reuse connections per user
- Implement connection timeouts

**Caching**:
- Cache user tokens in memory (with TTL)
- Cache Notion metadata per user
- Implement cache invalidation strategy

### Security

**Token Encryption**:
- ✅ Use Fernet encryption (symmetric)
- ✅ Rotate encryption keys regularly
- ✅ Store keys in secure vault (AWS KMS, HashiCorp Vault)

**Access Control**:
- Implement rate limiting per user
- Log all token access
- Monitor for suspicious activity

**Token Validation**:
- Validate tokens on registration
- Re-validate periodically
- Handle revoked tokens gracefully

### Monitoring

Track metrics:
- Active users per service
- Token validation failures
- API errors per user
- Usage patterns per tenant

## Database Schema

For production, use a proper database instead of encrypted files:

```sql
CREATE TABLE user_tokens (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    service VARCHAR(50) NOT NULL,
    encrypted_token TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_validated TIMESTAMP,
    is_valid BOOLEAN DEFAULT true,
    UNIQUE(user_id, service)
);

CREATE INDEX idx_user_service ON user_tokens(user_id, service);
CREATE INDEX idx_last_validated ON user_tokens(last_validated);
```

## Cost Analysis

### Single Token vs BYOT

**Single Token**:
- Infrastructure: $X/month (one instance)
- Limitations: Shared rate limits, no isolation
- Scales to: ~100 users before bottleneck

**BYOT**:
- Infrastructure: $X/month (orchestrator) + $Y per 1000 users
- Benefits: Independent rate limits, full isolation
- Scales to: Millions of users (linear scaling)

### ROI Calculation

```
Break-even point = (BYOT fixed cost - Single token cost) / (Cost per user * users)

Example:
- Single token: $100/month (max 100 users)
- BYOT: $200/month + $1 per 1000 users
- Break-even: 100 users
- At 1000 users: BYOT saves $800/month
```

## Migration Path

### From Single Token to BYOT

1. **Phase 1: Add BYOT support** (this guide)
   - Implement TokenManager
   - Update agents for user tokens
   - Keep env var fallback

2. **Phase 2: Migrate existing users**
   - Export user list
   - Email migration instructions
   - Set deadline for migration

3. **Phase 3: Deprecate shared token**
   - Remove NOTION_TOKEN from .env
   - All users on their own tokens
   - Remove fallback code

## Testing Strategy

### Unit Tests

```python
import pytest
from connectors.token_manager import TokenManager

def test_token_storage():
    tm = TokenManager()
    tm.set_token("user1", "notion", "secret_test")
    assert tm.get_token("user1", "notion") == "secret_test"

def test_token_encryption():
    tm = TokenManager()
    tm.set_token("user1", "notion", "secret_test")
    # Tokens should be encrypted on disk
    with open(tm.storage_file, 'rb') as f:
        data = f.read()
    assert b"secret_test" not in data  # Should be encrypted
```

### Integration Tests

```python
async def test_byot_notion_agent():
    # Register test token
    token_manager = get_token_manager()
    token_manager.set_token("test_user", "notion", os.getenv("TEST_NOTION_TOKEN"))

    # Initialize agent with user context
    agent = NotionAgent(user_id="test_user", verbose=True)
    await agent.initialize()

    # Test operation
    result = await agent.execute("List my pages")
    assert "Found" in result

    await agent.cleanup()
```

## Production Checklist

- [ ] Generate and secure encryption key
- [ ] Set up token storage (database or encrypted files)
- [ ] Implement user registration flow
- [ ] Add token validation
- [ ] Set up monitoring and logging
- [ ] Create user documentation
- [ ] Test with multiple users
- [ ] Implement rate limiting
- [ ] Set up backup/recovery for tokens
- [ ] Configure alerts for token failures
- [ ] Plan for token rotation
- [ ] Audit security implementation

## Future Enhancements

1. **Token Rotation**: Auto-rotate tokens periodically
2. **OAuth Flow**: Let users authenticate via OAuth instead of manual token
3. **Team Management**: Support for team/org-wide tokens
4. **Usage Analytics**: Per-user usage tracking and billing
5. **Token Health Monitoring**: Auto-detect and alert on invalid tokens
6. **Multi-Region Support**: Token storage across regions for compliance

## Support & Resources

- **Notion API**: https://developers.notion.com/
- **Cryptography Docs**: https://cryptography.io/
- **MCP Protocol**: https://github.com/anthropics/mcp
- **Security Best Practices**: OWASP guidelines

---

**Ready to scale?** Follow this guide step-by-step to implement BYOT and unlock multi-tenant capabilities!
