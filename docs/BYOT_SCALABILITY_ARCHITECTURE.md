# Project Aerius: BYOT Multi-Tenant Scalability Architecture

**Author**: Senior Architecture Team
**Date**: 2025-11-14
**Status**: Architectural Design Document
**Version**: 1.0

---

## Executive Summary

This document outlines the comprehensive architectural transformation required to convert Project Aerius from a **single-user CLI tool** to a **scalable, multi-tenant web service** with BYOT (Bring Your Own Token) support. This transformation addresses:

- **Multi-tenancy**: Support thousands of concurrent users with isolated sessions
- **BYOT**: Allow users to connect their own Slack, GitHub, Jira, Notion credentials
- **Scalability**: Horizontal scaling using Kafka-based message queuing and load balancing
- **Data Persistence**: Transition from file-based to database-backed storage
- **Security**: Token encryption, tenant isolation, and audit logging

**Estimated Complexity**: 6-9 months with 3-4 senior engineers
**Risk Level**: HIGH (complete architectural redesign)

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Critical Challenges & Honest Assessment](#2-critical-challenges--honest-assessment)
3. [Target Architecture Overview](#3-target-architecture-overview)
4. [BYOT Implementation Strategy](#4-byot-implementation-strategy)
5. [Kafka Message Queue Architecture](#5-kafka-message-queue-architecture)
6. [Load Balancing & Horizontal Scaling](#6-load-balancing--horizontal-scaling)
7. [Multi-Tenant Data Storage](#7-multi-tenant-data-storage)
8. [MCP Connection Management](#8-mcp-connection-management)
9. [Security Architecture](#9-security-architecture)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [Cost Analysis & Trade-offs](#11-cost-analysis--trade-offs)
12. [Migration Strategy](#12-migration-strategy)

---

## 1. Current Architecture Analysis

### 1.1 What We Have Today

```
┌─────────────────────────────────────────────────┐
│  Single User CLI Session                        │
│  - One process per user                         │
│  - Environment variables for credentials        │
│  - File-based storage (JSON)                    │
│  - No persistence between sessions              │
└─────────────────────────────────────────────────┘
         │
         ├─ 8 Agents (4 MCP-based, 4 direct)
         ├─ MCP subprocesses (npx stdio)
         ├─ Session logs (logs/*.json)
         └─ Workspace knowledge (data/*.json)
```

**Strengths**:
- ✅ Modular agent architecture
- ✅ MCP protocol integration (industry standard)
- ✅ Async/await foundation (asyncio)
- ✅ Retry/circuit breaker patterns
- ✅ Session logging infrastructure

**Limitations for Multi-Tenancy**:
- ❌ No web server or API layer
- ❌ Single-user credential model (env vars)
- ❌ No database (file-based storage)
- ❌ No session persistence
- ❌ No user authentication/authorization
- ❌ MCP processes spawned per-session (resource intensive)
- ❌ No message queue for async processing
- ❌ No horizontal scaling capability

---

## 2. Critical Challenges & Honest Assessment

### 2.1 Architectural Transformation Scope

**This is not an incremental change**. We're redesigning the entire system:

| Component | Current | Required | Effort |
|-----------|---------|----------|--------|
| **Entry Point** | CLI (main.py) | FastAPI Web Server | Medium |
| **Authentication** | None | OAuth2 + JWT + BYOT | High |
| **Storage** | JSON files | PostgreSQL + Redis | High |
| **Session Mgmt** | Ephemeral (process) | Persistent (DB + cache) | High |
| **MCP Connections** | Subprocess per session | Connection pool | Very High |
| **Scalability** | Single process | Kafka + worker pool | Very High |
| **Credential Mgmt** | Env vars | Encrypted vault (KMS) | High |
| **Monitoring** | Basic logging | Full observability stack | Medium |

### 2.2 Hardest Problems to Solve

#### Problem 1: MCP Connection Pooling (VERY HARD)

**Current**: Each session spawns subprocess: `npx @modelcontextprotocol/server-slack`
- **Cost**: ~200MB RAM + Node.js process per user per MCP agent
- **Scale**: 1,000 concurrent users = 4,000 MCP processes (4 MCP agents) = **~800GB RAM**

**Challenge**: MCP protocol uses stdio (stdin/stdout) which is **1:1 process-to-session**. Cannot easily share.

**Options**:
1. **Run MCP as HTTP/SSE server** (requires MCP server modifications or wrapper)
   - Pros: Shareable, scalable
   - Cons: MCP servers don't natively support this, need custom wrapper

2. **Pool MCP processes per credential set** (not per user)
   - Pros: Reuse if 10 users share same Slack workspace
   - Cons: Complex state management, credential isolation issues

3. **Serverless MCP** (spawn on-demand, aggressive pooling)
   - Pros: Cost-effective at scale
   - Cons: Cold start latency (1-2s)

**Recommendation**: Build MCP HTTP adapter layer (Option 1) - **4-6 weeks dev time**

#### Problem 2: BYOT Token Management (HIGH COMPLEXITY)

**Users will provide**:
- `SLACK_BOT_TOKEN` + `SLACK_TEAM_ID`
- `GITHUB_PERSONAL_ACCESS_TOKEN`
- `JIRA_API_TOKEN` + `JIRA_EMAIL` + `JIRA_DOMAIN`
- `NOTION_TOKEN`
- `GOOGLE_API_KEY` (for Gemini LLM)

**Challenges**:
- **Token validation**: How do we verify tokens without executing requests?
- **Scopes/permissions**: Users may provide tokens with insufficient scopes
- **Token expiry**: Need refresh token flows (OAuth2)
- **Rotation**: Handle token revocation gracefully
- **Encryption**: NEVER store tokens plaintext

**Recommendation**: Implement OAuth2 flow + encrypted vault (AWS KMS / HashiCorp Vault)

#### Problem 3: Multi-Tenant Data Isolation (CRITICAL)

**Risks**:
- Tenant A sees Tenant B's chat history
- Tenant A uses Tenant B's Slack credentials
- Session leakage across users

**Mitigation**:
- **Row-Level Security (RLS)** in PostgreSQL
- **Tenant ID in every table** (composite primary keys)
- **Request context isolation** (using Python contextvars)
- **Encryption at rest** for all PII/tokens
- **Audit logging** for compliance

#### Problem 4: LLM API Key Management

**Current**: Single `GOOGLE_API_KEY` for all users

**BYOT Model**: Each user provides their own API key

**Implications**:
- **Rate limiting**: Per-user quotas (not global)
- **Cost attribution**: Track usage per tenant
- **Fallback**: What if user's key is invalid/exhausted?
- **Provider flexibility**: Support multiple LLM providers (OpenAI, Anthropic, Gemini)

**Recommendation**: Implement LLM router with per-tenant quota tracking

---

## 3. Target Architecture Overview

### 3.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER (nginx/ALB)                    │
│                     SSL Termination, Rate Limiting                   │
└───────────────┬──────────────────────────────────────────────────────┘
                │
    ┌───────────▼────────────┐      ┌──────────────────┐
    │  FastAPI Web Service   │      │  FastAPI Web... │ (N instances)
    │  - REST API            │      │                  │
    │  - WebSocket (chat)    │◄─────┤  Stateless       │
    │  - Authentication      │      │  Horizontal      │
    │  - Session management  │      │  Auto-scaling    │
    └───────┬────────────────┘      └──────────────────┘
            │
            ├──► Redis (Session Cache, Request Dedup)
            │
            ├──► PostgreSQL (Users, Sessions, Tokens, Audit)
            │
            └──► Kafka Producer (Async Agent Tasks)
                        │
        ┌───────────────▼────────────────────┐
        │        Kafka Message Broker        │
        │  Topics:                           │
        │  - agent.tasks.high_priority       │
        │  - agent.tasks.normal              │
        │  - agent.tasks.low_priority        │
        │  - agent.results                   │
        │  - agent.status_updates            │
        └───────────────┬────────────────────┘
                        │
        ┌───────────────▼────────────────────┐
        │     Kafka Consumer Workers         │
        │  (Agent Execution Pool)            │
        │                                    │
        │  ┌──────────────────────────┐     │
        │  │  Worker Pod 1            │     │ (N workers)
        │  │  - Pull from Kafka       │     │
        │  │  - Execute agent tasks   │     │
        │  │  - Manage MCP connections│     │
        │  │  - Publish results       │     │
        │  └──────────────────────────┘     │
        │                                    │
        │  ┌──────────────────────────┐     │
        │  │  MCP Connection Pool     │     │
        │  │  - HTTP/SSE MCP Adapters │     │
        │  │  - Per-credential pools  │     │
        │  │  - Lifecycle management  │     │
        │  └──────────────────────────┘     │
        └────────────────────────────────────┘
                        │
                        ├──► S3 (Session Logs, Analytics)
                        ├──► Vault (Encrypted Tokens - KMS)
                        └──► CloudWatch/Prometheus (Metrics)
```

### 3.2 Request Flow: User Sends Message

```
1. User (Browser/App)
   │
   └──► POST /api/v1/chat/sessions/{session_id}/messages
        {
          "message": "Check my GitHub PRs",
          "context": {...}
        }
        │
        ▼
2. Load Balancer (nginx)
   │ - SSL termination
   │ - Rate limiting (per-user)
   └──► FastAPI Instance #3
        │
        ▼
3. FastAPI Middleware
   │ - JWT validation
   │ - Extract tenant_id, user_id
   │ - Set request context
   │
   ▼
4. Session Manager
   │ - Validate session exists
   │ - Load session from PostgreSQL
   │ - Check user owns session (authz)
   │
   ▼
5. Intelligence Pipeline
   │ - Classify intent (which agent?)
   │ - Decompose task
   │ - Build execution plan
   │
   ▼
6. Kafka Producer
   │ - Publish task to Kafka topic
   │ - Topic: agent.tasks.normal
   │ - Message:
   │   {
   │     "task_id": "uuid",
   │     "tenant_id": "tenant-123",
   │     "session_id": "session-456",
   │     "agent": "github",
   │     "instruction": "list PRs",
   │     "credentials_ref": "cred-789",  // NOT the actual token
   │     "priority": "normal"
   │   }
   │
   ▼
7. FastAPI Response (immediate)
   │ - 202 Accepted
   │ - { "task_id": "uuid", "status": "queued" }
   │ - Client polls or uses WebSocket
   │
   └──► User sees "Processing..." in UI
```

### 3.3 Worker Processing Flow

```
1. Kafka Consumer (Worker Pod #7)
   │ - Poll topic: agent.tasks.normal
   │ - Consume message
   │
   ▼
2. Task Validator
   │ - Verify tenant_id exists
   │ - Check rate limits
   │ - Load credentials from Vault
   │
   ▼
3. Credential Resolver
   │ - Fetch from Vault: credentials_ref="cred-789"
   │ - Decrypt using KMS
   │ - Returns: { "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..." }
   │
   ▼
4. MCP Connection Pool
   │ - Get or create MCP connection for:
   │   - Agent: github
   │   - Credentials hash: sha256(token)
   │ - Reuse if already exists
   │ - Create new if not found
   │
   ▼
5. Agent Execution (with retry/circuit breaker)
   │ - GitHub Agent runs
   │ - Calls MCP tools
   │ - Returns result
   │
   ▼
6. Result Publisher
   │ - Publish to Kafka: agent.results
   │ - Message:
   │   {
   │     "task_id": "uuid",
   │     "status": "completed",
   │     "result": { "prs": [...] },
   │     "execution_time_ms": 1250
   │   }
   │ - Store in PostgreSQL: task_results table
   │ - Update Redis cache
   │
   ▼
7. WebSocket Notifier
   │ - Find active WebSocket connections for session
   │ - Push result to client
   │
   └──► User sees result in real-time
```

---

## 4. BYOT Implementation Strategy

### 4.1 BYOT Onboarding Flow

```
┌─────────────────────────────────────────────────────┐
│  Step 1: User Registration                          │
│  - Email + Password OR OAuth (Google/GitHub)        │
│  - Create tenant_id (for org accounts)              │
│  - Generate JWT access token                        │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Integration Setup (BYOT)                   │
│                                                      │
│  UI: "Connect Your Tools"                           │
│                                                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  [+] Connect Slack                           │  │
│  │      Enter Bot Token & Team ID               │  │
│  │      [Validate] [Save]                       │  │
│  ├──────────────────────────────────────────────┤  │
│  │  [+] Connect GitHub                          │  │
│  │      [OAuth Flow] OR [PAT Token]             │  │
│  ├──────────────────────────────────────────────┤  │
│  │  [+] Connect Jira                            │  │
│  │  [+] Connect Notion                          │  │
│  │  [+] Connect LLM Provider                    │  │
│  │      - OpenAI API Key                        │  │
│  │      - Anthropic API Key                     │  │
│  │      - Google Gemini API Key                 │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Token Validation & Storage                 │
│                                                      │
│  Backend Process:                                   │
│  1. Validate token (test API call)                  │
│     - Slack: slack.auth.test                        │
│     - GitHub: GET /user                             │
│     - Jira: GET /myself                             │
│                                                      │
│  2. Detect scopes/permissions                       │
│     - Warning if insufficient scopes                │
│                                                      │
│  3. Encrypt token                                   │
│     - KMS encryption (AWS KMS / HashiCorp Vault)    │
│     - Never store plaintext                         │
│                                                      │
│  4. Store in database                               │
│     - user_credentials table                        │
│     - Columns:                                      │
│       - credential_id (UUID)                        │
│       - tenant_id                                   │
│       - user_id                                     │
│       - integration_type (slack, github, etc.)      │
│       - encrypted_token (bytea)                     │
│       - token_metadata (scopes, expires_at)         │
│       - created_at, updated_at                      │
│       - is_active (bool)                            │
└─────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: Ready to Use                               │
│  - User can now chat with agents                    │
│  - Agents use BYOT credentials                      │
└─────────────────────────────────────────────────────┘
```

### 4.2 Database Schema: user_credentials

```sql
CREATE TABLE user_credentials (
    credential_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Integration details
    integration_type VARCHAR(50) NOT NULL, -- 'slack', 'github', 'jira', 'notion', 'openai', 'gemini'
    integration_name VARCHAR(255), -- User-friendly name: "My GitHub", "Work Slack"

    -- Encrypted credentials
    encrypted_credentials BYTEA NOT NULL, -- Encrypted JSON blob
    encryption_key_id VARCHAR(255) NOT NULL, -- KMS key ID used for encryption

    -- Token metadata (NOT encrypted - for display/validation)
    token_metadata JSONB, -- { "scopes": ["repo", "user"], "expires_at": "2025-12-31", "workspace": "acme-corp" }

    -- Status
    is_active BOOLEAN DEFAULT true,
    last_validated_at TIMESTAMP,
    validation_status VARCHAR(50), -- 'valid', 'invalid', 'expired', 'insufficient_scopes'
    validation_error TEXT,

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by UUID REFERENCES users(user_id),

    -- Indexes
    CONSTRAINT unique_integration_per_user UNIQUE (tenant_id, user_id, integration_type, integration_name)
);

CREATE INDEX idx_credentials_tenant_user ON user_credentials(tenant_id, user_id);
CREATE INDEX idx_credentials_active ON user_credentials(is_active) WHERE is_active = true;
```

### 4.3 Credential Encryption/Decryption

```python
# New file: core/credential_vault.py

import json
import base64
from typing import Dict, Optional
from cryptography.fernet import Fernet
import boto3  # For AWS KMS

class CredentialVault:
    """Secure credential management using AWS KMS encryption"""

    def __init__(self, kms_client: boto3.client, key_id: str):
        self.kms = kms_client
        self.key_id = key_id

    async def encrypt_credentials(self, credentials: Dict[str, str]) -> tuple[bytes, str]:
        """
        Encrypt credentials using envelope encryption:
        1. Generate data key from KMS
        2. Use data key to encrypt credentials (Fernet)
        3. Return encrypted credentials + encrypted data key
        """
        # Generate data key
        response = self.kms.generate_data_key(
            KeyId=self.key_id,
            KeySpec='AES_256'
        )

        plaintext_key = response['Plaintext']
        encrypted_key = base64.b64encode(response['CiphertextBlob']).decode()

        # Encrypt credentials with data key
        fernet = Fernet(base64.urlsafe_b64encode(plaintext_key[:32]))
        credentials_json = json.dumps(credentials).encode()
        encrypted_credentials = fernet.encrypt(credentials_json)

        return encrypted_credentials, encrypted_key

    async def decrypt_credentials(
        self,
        encrypted_credentials: bytes,
        encrypted_key: str
    ) -> Dict[str, str]:
        """Decrypt credentials using KMS"""
        # Decrypt data key
        response = self.kms.decrypt(
            CiphertextBlob=base64.b64decode(encrypted_key)
        )
        plaintext_key = response['Plaintext']

        # Decrypt credentials
        fernet = Fernet(base64.urlsafe_b64encode(plaintext_key[:32]))
        decrypted_json = fernet.decrypt(encrypted_credentials)

        return json.loads(decrypted_json.decode())

    async def validate_token(
        self,
        integration_type: str,
        credentials: Dict[str, str]
    ) -> tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate token by making test API call
        Returns: (is_valid, error_message, metadata)
        """
        if integration_type == 'slack':
            return await self._validate_slack(credentials)
        elif integration_type == 'github':
            return await self._validate_github(credentials)
        elif integration_type == 'jira':
            return await self._validate_jira(credentials)
        # ... etc

    async def _validate_slack(self, creds: Dict) -> tuple[bool, Optional[str], Optional[Dict]]:
        """Test Slack token validity"""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://slack.com/api/auth.test',
                    headers={'Authorization': f"Bearer {creds['SLACK_BOT_TOKEN']}"}
                ) as response:
                    data = await response.json()

                    if data.get('ok'):
                        metadata = {
                            'workspace': data.get('team'),
                            'user': data.get('user'),
                            'scopes': data.get('scopes', [])
                        }
                        return True, None, metadata
                    else:
                        return False, data.get('error', 'Unknown error'), None
        except Exception as e:
            return False, str(e), None
```

### 4.4 Integration Setup API Endpoints

```python
# New file: api/v1/integrations.py

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

router = APIRouter(prefix="/api/v1/integrations", tags=["integrations"])

class AddCredentialRequest(BaseModel):
    integration_type: str  # 'slack', 'github', etc.
    integration_name: Optional[str]  # "Work Slack"
    credentials: Dict[str, str]  # { "SLACK_BOT_TOKEN": "xoxb-...", "SLACK_TEAM_ID": "T..." }

class CredentialResponse(BaseModel):
    credential_id: str
    integration_type: str
    integration_name: Optional[str]
    is_active: bool
    validation_status: str
    token_metadata: Optional[Dict]
    created_at: str

@router.post("/credentials", response_model=CredentialResponse)
async def add_credential(
    request: AddCredentialRequest,
    current_user: User = Depends(get_current_user),
    vault: CredentialVault = Depends(get_vault),
    db: Database = Depends(get_db)
):
    """
    Add new integration credentials (BYOT)
    """
    # 1. Validate token
    is_valid, error, metadata = await vault.validate_token(
        request.integration_type,
        request.credentials
    )

    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid credentials: {error}"
        )

    # 2. Encrypt credentials
    encrypted_creds, encryption_key_id = await vault.encrypt_credentials(
        request.credentials
    )

    # 3. Store in database
    credential = await db.user_credentials.create({
        'tenant_id': current_user.tenant_id,
        'user_id': current_user.user_id,
        'integration_type': request.integration_type,
        'integration_name': request.integration_name,
        'encrypted_credentials': encrypted_creds,
        'encryption_key_id': encryption_key_id,
        'token_metadata': metadata,
        'validation_status': 'valid',
        'last_validated_at': datetime.utcnow()
    })

    # 4. Audit log
    await db.audit_log.create({
        'tenant_id': current_user.tenant_id,
        'user_id': current_user.user_id,
        'action': 'credential.created',
        'resource_type': 'user_credential',
        'resource_id': credential.credential_id,
        'metadata': {'integration_type': request.integration_type}
    })

    return CredentialResponse.from_orm(credential)

@router.get("/credentials", response_model=list[CredentialResponse])
async def list_credentials(
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """List all integrations for current user"""
    credentials = await db.user_credentials.find_many(
        where={
            'tenant_id': current_user.tenant_id,
            'user_id': current_user.user_id,
            'is_active': True
        }
    )
    return [CredentialResponse.from_orm(c) for c in credentials]

@router.delete("/credentials/{credential_id}")
async def delete_credential(
    credential_id: str,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """Revoke/delete integration"""
    # Verify ownership
    credential = await db.user_credentials.find_unique(
        where={'credential_id': credential_id}
    )

    if not credential or credential.user_id != current_user.user_id:
        raise HTTPException(status_code=404, detail="Credential not found")

    # Soft delete
    await db.user_credentials.update(
        where={'credential_id': credential_id},
        data={'is_active': False, 'updated_at': datetime.utcnow()}
    )

    # Audit
    await db.audit_log.create({
        'action': 'credential.deleted',
        'resource_id': credential_id
    })

    return {"status": "deleted"}
```

---

## 5. Kafka Message Queue Architecture

### 5.1 Why Kafka for This Use Case?

**Requirements**:
1. **Async processing**: User shouldn't wait for slow agent tasks (30-120s)
2. **Horizontal scaling**: Run 10-100+ worker pods
3. **Priority queues**: High-priority tasks (e.g., user queries) before low-priority (e.g., batch analytics)
4. **Reliability**: At-least-once delivery, retry failed tasks
5. **Observability**: Track task status, execution times, errors

**Kafka Benefits**:
- ✅ High throughput (millions of messages/sec)
- ✅ Horizontal scalability (partition-based)
- ✅ Durability (replicated, persistent)
- ✅ Consumer groups (load balancing across workers)
- ✅ Message ordering within partitions
- ✅ Built-in monitoring (Kafka metrics)

**Alternatives Considered**:
- **RabbitMQ**: Simpler, but lower throughput (10-100k msg/sec)
- **AWS SQS**: Serverless, but no ordering guarantees, higher latency
- **Redis Streams**: Fast, but less durable than Kafka
- **Celery + Redis**: Python-native, but less scalable than Kafka

**Verdict**: **Kafka is the right choice** for enterprise-scale multi-tenant system.

### 5.2 Kafka Topics Design

```
┌────────────────────────────────────────────────────────────┐
│  Topic: agent.tasks.high_priority                          │
│  - User-facing queries (response expected < 5s)            │
│  - Partitions: 20 (for parallelism)                        │
│  - Retention: 1 hour (tasks should complete quickly)       │
│  - Consumers: 20 worker pods (1 per partition)             │
│  - Examples: "Check my GitHub PRs", "Send Slack message"   │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Topic: agent.tasks.normal                                 │
│  - Standard agent tasks (response expected < 30s)          │
│  - Partitions: 50                                          │
│  - Retention: 6 hours                                      │
│  - Consumers: 50 worker pods                               │
│  - Examples: "Search Jira tickets", "Analyze code"         │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Topic: agent.tasks.low_priority                           │
│  - Background tasks, batch processing                      │
│  - Partitions: 10                                          │
│  - Retention: 24 hours                                     │
│  - Consumers: 10 worker pods                               │
│  - Examples: "Generate weekly report", "Sync all repos"    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Topic: agent.results                                      │
│  - Task results published by workers                       │
│  - Partitions: 20                                          │
│  - Retention: 24 hours                                     │
│  - Consumers: FastAPI instances (for WebSocket push)       │
│  - Message: { task_id, status, result, error, timing }    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Topic: agent.status_updates                               │
│  - Intermediate status updates (progress %, logs)          │
│  - Partitions: 10                                          │
│  - Retention: 1 hour                                       │
│  - Consumers: FastAPI instances (WebSocket push)           │
│  - Message: { task_id, progress: 0.75, message: "..." }   │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│  Topic: audit.events                                       │
│  - Audit trail for compliance (immutable log)              │
│  - Partitions: 5                                           │
│  - Retention: 90 days (compliance requirement)             │
│  - Consumers: Analytics pipeline, SIEM integration         │
│  - Message: { user_id, action, resource, timestamp, ... } │
└────────────────────────────────────────────────────────────┘
```

### 5.3 Message Schema (Avro)

```json
// agent_task_v1.avsc
{
  "type": "record",
  "name": "AgentTask",
  "namespace": "com.aerius.agent",
  "fields": [
    {"name": "task_id", "type": "string"},
    {"name": "tenant_id", "type": "string"},
    {"name": "user_id", "type": "string"},
    {"name": "session_id", "type": "string"},

    {"name": "agent_name", "type": "string"},
    {"name": "instruction", "type": "string"},
    {"name": "context", "type": ["null", "string"], "default": null},

    {"name": "credential_id", "type": "string"},  // Reference to user_credentials
    {"name": "llm_provider", "type": "string"},   // 'gemini', 'openai', 'anthropic'
    {"name": "llm_credential_id", "type": "string"},

    {"name": "priority", "type": "int", "default": 5},  // 1-10
    {"name": "timeout_seconds", "type": "int", "default": 120},

    {"name": "created_at", "type": "long"},  // Unix timestamp (ms)
    {"name": "retry_count", "type": "int", "default": 0},
    {"name": "max_retries", "type": "int", "default": 3},

    {"name": "idempotency_key", "type": "string"},  // For deduplication
    {"name": "callback_url", "type": ["null", "string"], "default": null}
  ]
}
```

### 5.4 Kafka Producer (FastAPI)

```python
# New file: infrastructure/kafka_producer.py

from confluent_kafka import Producer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
import json
import time

class AgentTaskProducer:
    """Publishes agent tasks to Kafka"""

    def __init__(self, kafka_config: dict, schema_registry_url: str):
        self.producer = Producer({
            'bootstrap.servers': kafka_config['bootstrap_servers'],
            'client.id': 'aerius-api',
            'compression.type': 'snappy',
            'linger.ms': 10,  # Batch messages for 10ms
            'batch.size': 32768,  # 32KB batches
            'acks': 'all',  # Wait for all replicas
            'retries': 5
        })

        # Schema registry for Avro serialization
        schema_registry = SchemaRegistryClient({'url': schema_registry_url})
        with open('schemas/agent_task_v1.avsc') as f:
            schema_str = f.read()
        self.serializer = AvroSerializer(schema_registry, schema_str)

    async def publish_task(
        self,
        task: dict,
        priority: str = 'normal'
    ) -> str:
        """
        Publish agent task to Kafka

        Args:
            task: Task payload (matches AgentTask schema)
            priority: 'high', 'normal', or 'low'

        Returns:
            task_id
        """
        topic = f"agent.tasks.{priority}_priority"

        # Add metadata
        task['created_at'] = int(time.time() * 1000)
        task['retry_count'] = 0

        # Serialize with Avro
        value = self.serializer(task, None)

        # Use tenant_id as partition key (ensures tenant affinity)
        key = task['tenant_id'].encode('utf-8')

        # Publish
        self.producer.produce(
            topic=topic,
            key=key,
            value=value,
            callback=self._delivery_callback
        )

        # Don't block - async publish
        self.producer.poll(0)

        return task['task_id']

    def _delivery_callback(self, err, msg):
        """Called when message is delivered or fails"""
        if err:
            # Log error, send to dead letter queue
            print(f"Message delivery failed: {err}")
        else:
            # Success
            print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    def flush(self):
        """Wait for all messages to be delivered"""
        self.producer.flush(timeout=30)
```

### 5.5 Kafka Consumer (Worker)

```python
# New file: workers/agent_task_consumer.py

from confluent_kafka import Consumer, KafkaError
from confluent_kafka.schema_registry.avro import AvroDeserializer
import asyncio
import signal

class AgentTaskConsumer:
    """Consumes agent tasks from Kafka and executes them"""

    def __init__(
        self,
        kafka_config: dict,
        consumer_group: str = 'agent-workers',
        topics: list = None
    ):
        self.consumer = Consumer({
            'bootstrap.servers': kafka_config['bootstrap_servers'],
            'group.id': consumer_group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,  # Manual commit for reliability
            'max.poll.interval.ms': 300000,  # 5 minutes (long processing)
            'session.timeout.ms': 60000  # 1 minute
        })

        self.topics = topics or [
            'agent.tasks.high_priority',
            'agent.tasks.normal',
            'agent.tasks.low_priority'
        ]

        self.consumer.subscribe(self.topics)
        self.running = True

        # Initialize agent executor, credential vault, etc.
        self.executor = AgentExecutor()
        self.vault = CredentialVault()
        self.result_publisher = ResultPublisher()

    async def start(self):
        """Start consuming messages"""
        print(f"Worker started, consuming from {self.topics}")

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        while self.running:
            # Poll for messages (timeout 1s)
            msg = self.consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    continue

            # Process message
            try:
                task = self._deserialize_task(msg.value())
                result = await self._process_task(task)

                # Publish result
                await self.result_publisher.publish(result)

                # Commit offset (message processed successfully)
                self.consumer.commit(asynchronous=False)

            except Exception as e:
                print(f"Task processing failed: {e}")
                # Don't commit - message will be redelivered
                # Send to dead letter queue after max retries
                await self._handle_failure(task, e)

    async def _process_task(self, task: dict) -> dict:
        """Execute agent task"""
        task_id = task['task_id']

        print(f"Processing task {task_id}: {task['agent_name']}")

        # 1. Load credentials
        credentials = await self.vault.decrypt_credentials(
            credential_id=task['credential_id']
        )

        llm_credentials = await self.vault.decrypt_credentials(
            credential_id=task['llm_credential_id']
        )

        # 2. Execute agent with timeout
        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                self.executor.execute_agent(
                    agent_name=task['agent_name'],
                    instruction=task['instruction'],
                    credentials=credentials,
                    llm_credentials=llm_credentials,
                    context=task.get('context')
                ),
                timeout=task['timeout_seconds']
            )

            execution_time = time.time() - start_time

            return {
                'task_id': task_id,
                'status': 'completed',
                'result': result,
                'execution_time_ms': int(execution_time * 1000),
                'worker_id': self.worker_id,
                'completed_at': int(time.time() * 1000)
            }

        except asyncio.TimeoutError:
            return {
                'task_id': task_id,
                'status': 'timeout',
                'error': f"Task exceeded timeout of {task['timeout_seconds']}s"
            }
        except Exception as e:
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__
            }

    def _shutdown_handler(self, signum, frame):
        """Graceful shutdown on SIGTERM"""
        print("Shutdown signal received, finishing current task...")
        self.running = False

    def close(self):
        """Close consumer"""
        self.consumer.close()
```

### 5.6 Dead Letter Queue (DLQ) Pattern

```python
# Handle failed messages after max retries

async def _handle_failure(self, task: dict, error: Exception):
    """Send failed task to dead letter queue"""

    if task['retry_count'] >= task['max_retries']:
        # Max retries exceeded - send to DLQ
        dlq_message = {
            'original_task': task,
            'error': str(error),
            'error_type': type(error).__name__,
            'failed_at': int(time.time() * 1000),
            'worker_id': self.worker_id
        }

        await self.producer.publish(
            topic='agent.tasks.dead_letter',
            message=dlq_message
        )

        # Alert operations team
        await self.alerting.send_alert(
            severity='high',
            message=f"Task {task['task_id']} sent to DLQ after {task['retry_count']} retries"
        )

        # Commit offset (don't retry further)
        self.consumer.commit()
    else:
        # Retry - republish with incremented retry_count
        task['retry_count'] += 1
        await self.producer.publish_task(
            task=task,
            priority=task['priority']
        )
        self.consumer.commit()
```

---

## 6. Load Balancing & Horizontal Scaling

### 6.1 Architecture Layers

```
                     Internet
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Layer 7 Load Balancer       │
        │   (AWS ALB / nginx)            │
        │   - SSL termination            │
        │   - Path-based routing         │
        │   - WebSocket support          │
        │   - Rate limiting (global)     │
        └────┬──────────────┬────────────┘
             │              │
    ┌────────▼──────┐  ┌───▼────────────┐
    │ FastAPI Pod 1 │  │ FastAPI Pod N  │ (Auto-scaling: 5-50 pods)
    │               │  │                 │
    │ - Stateless   │  │ - Stateless     │
    │ - JWT auth    │  │ - JWT auth      │
    │ - Kafka pub   │  │ - Kafka pub     │
    └────┬──────────┘  └───┬────────────┘
         │                  │
         └──────┬───────────┘
                │
      ┌─────────▼──────────┐
      │   Shared State     │
      ├────────────────────┤
      │ - PostgreSQL (RDS) │
      │ - Redis (ElastiCache)│
      │ - Kafka (MSK)      │
      └────────────────────┘
```

### 6.2 Auto-Scaling Configuration

#### FastAPI Web Service (Kubernetes HPA)

```yaml
# k8s/fastapi-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: aerius-api
spec:
  replicas: 5  # Minimum
  selector:
    matchLabels:
      app: aerius-api
  template:
    metadata:
      labels:
        app: aerius-api
    spec:
      containers:
      - name: fastapi
        image: aerius/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-broker-1:9092,kafka-broker-2:9092"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aerius-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aerius-api
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"  # 1000 req/s per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50  # Scale up by 50% at a time
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Pods
        value: 2  # Remove max 2 pods at a time
        periodSeconds: 120
```

#### Kafka Consumer Workers (Kubernetes HPA)

```yaml
# k8s/worker-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: aerius-agent-worker
spec:
  replicas: 20  # Start with 20 workers
  template:
    spec:
      containers:
      - name: worker
        image: aerius/worker:latest
        env:
        - name: KAFKA_CONSUMER_GROUP
          value: "agent-workers"
        - name: KAFKA_TOPICS
          value: "agent.tasks.high_priority,agent.tasks.normal,agent.tasks.low_priority"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aerius-worker-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: aerius-agent-worker
  minReplicas: 20
  maxReplicas: 200
  metrics:
  - type: External
    external:
      metric:
        name: kafka_consumer_lag
        selector:
          matchLabels:
            topic: agent.tasks.normal
      target:
        type: AverageValue
        averageValue: "100"  # Scale up if lag > 100 messages per worker
  behavior:
    scaleUp:
      policies:
      - type: Percent
        value: 100  # Double workers if lagging
        periodSeconds: 60
```

### 6.3 Load Balancing Strategies

#### nginx Configuration

```nginx
# nginx.conf - Layer 7 Load Balancer

upstream fastapi_backend {
    # Least connections algorithm (best for long-running requests)
    least_conn;

    # Backend servers (Kubernetes service)
    server aerius-api-service:8000 max_fails=3 fail_timeout=30s;

    # Keepalive connections
    keepalive 32;
}

# Rate limiting zones
limit_req_zone $binary_remote_addr zone=user_limit:10m rate=10r/s;
limit_req_zone $http_x_tenant_id zone=tenant_limit:10m rate=100r/s;

server {
    listen 443 ssl http2;
    server_name api.aerius.com;

    # SSL configuration
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;

    # Rate limiting
    limit_req zone=user_limit burst=20 nodelay;
    limit_req zone=tenant_limit burst=200 nodelay;

    # API endpoints
    location /api/ {
        proxy_pass http://fastapi_backend;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Connection pooling
        proxy_set_header Connection "";
    }

    # WebSocket endpoint (for real-time updates)
    location /ws/ {
        proxy_pass http://fastapi_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Long timeout for WebSocket
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://fastapi_backend;
        access_log off;
    }
}
```

### 6.4 Session Affinity (Sticky Sessions)

**Problem**: WebSocket connections need to stay with same FastAPI instance

**Solution**: Use Redis-backed session management (not cookie-based stickiness)

```python
# infrastructure/websocket_manager.py

from fastapi import WebSocket
import redis.asyncio as redis
import json

class WebSocketManager:
    """Manages WebSocket connections across multiple FastAPI instances"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Register WebSocket connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket

        # Register in Redis (for cross-instance messaging)
        await self.redis.hset(
            'websocket_connections',
            session_id,
            json.dumps({
                'instance_id': INSTANCE_ID,  # This pod's ID
                'connected_at': time.time()
            })
        )

        # Subscribe to Redis pub/sub for this session
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f'session:{session_id}:updates')

        # Listen for messages from other instances
        asyncio.create_task(self._listen_redis_messages(session_id, websocket, pubsub))

    async def send_update(self, session_id: str, message: dict):
        """Send update to WebSocket (works across instances)"""

        # Check if connected to this instance
        if session_id in self.active_connections:
            ws = self.active_connections[session_id]
            await ws.send_json(message)
        else:
            # Connected to different instance - use Redis pub/sub
            await self.redis.publish(
                f'session:{session_id}:updates',
                json.dumps(message)
            )

    async def _listen_redis_messages(self, session_id: str, websocket: WebSocket, pubsub):
        """Listen for Redis pub/sub messages"""
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await websocket.send_json(data)
```

---

## 7. Multi-Tenant Data Storage

### 7.1 Database Schema (PostgreSQL)

```sql
-- Core multi-tenancy schema

-- 1. Tenants (organizations)
CREATE TABLE tenants (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_name VARCHAR(255) NOT NULL,
    tenant_slug VARCHAR(100) UNIQUE NOT NULL, -- 'acme-corp'

    -- Subscription info
    plan_type VARCHAR(50) NOT NULL DEFAULT 'free', -- 'free', 'pro', 'enterprise'
    max_users INT DEFAULT 5,
    max_sessions_per_month INT DEFAULT 100,

    -- Status
    is_active BOOLEAN DEFAULT true,
    suspended_at TIMESTAMP,
    suspension_reason TEXT,

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    -- Search
    tsv_tenant_name tsvector GENERATED ALWAYS AS (to_tsvector('english', tenant_name)) STORED
);

CREATE INDEX idx_tenants_active ON tenants(is_active) WHERE is_active = true;
CREATE INDEX idx_tenants_search ON tenants USING gin(tsv_tenant_name);

-- 2. Users
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,

    -- Identity
    email VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255), -- NULL if OAuth-only
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),

    -- OAuth
    oauth_provider VARCHAR(50), -- 'google', 'github', NULL
    oauth_provider_id VARCHAR(255),

    -- Role-based access control
    role VARCHAR(50) DEFAULT 'member', -- 'owner', 'admin', 'member', 'viewer'

    -- Status
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    last_login_at TIMESTAMP,

    -- Audit
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_email_per_tenant UNIQUE (tenant_id, email),
    CONSTRAINT unique_oauth UNIQUE (oauth_provider, oauth_provider_id)
);

CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);

-- 3. Sessions (chat sessions)
CREATE TABLE chat_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    -- Session metadata
    title VARCHAR(500), -- "Debugging API issue", auto-generated from first message
    session_type VARCHAR(50) DEFAULT 'interactive', -- 'interactive', 'batch', 'automation'

    -- Lifecycle
    started_at TIMESTAMP DEFAULT NOW(),
    ended_at TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT NOW(),
    message_count INT DEFAULT 0,

    -- Settings (per-session preferences)
    settings JSONB DEFAULT '{}'::jsonb,
    -- Example: {"llm_provider": "gemini", "temperature": 0.7, "enabled_agents": ["slack", "github"]}

    -- Status
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'paused', 'completed', 'failed'

    CONSTRAINT fk_tenant_user FOREIGN KEY (tenant_id, user_id) REFERENCES users(tenant_id, user_id)
);

CREATE INDEX idx_sessions_tenant_user ON chat_sessions(tenant_id, user_id);
CREATE INDEX idx_sessions_active ON chat_sessions(status, last_activity_at) WHERE status = 'active';

-- 4. Messages (chat messages within sessions)
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL, -- Denormalized for partitioning

    -- Message content
    role VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    content_type VARCHAR(50) DEFAULT 'text', -- 'text', 'markdown', 'code'

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    -- Example: {"agent": "github", "execution_time_ms": 1250, "tokens_used": 450}

    -- Ordering
    sequence_number INT NOT NULL, -- Message order within session

    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_message_sequence UNIQUE (session_id, sequence_number)
);

CREATE INDEX idx_messages_session ON messages(session_id, sequence_number);
CREATE INDEX idx_messages_tenant ON messages(tenant_id, created_at); -- For analytics

-- 5. Agent Tasks (tracking async task execution)
CREATE TABLE agent_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(message_id) ON DELETE SET NULL,

    -- Task details
    agent_name VARCHAR(100) NOT NULL,
    instruction TEXT NOT NULL,
    priority VARCHAR(20) DEFAULT 'normal',

    -- Execution
    status VARCHAR(50) DEFAULT 'queued', -- 'queued', 'running', 'completed', 'failed', 'timeout'
    worker_id VARCHAR(100), -- Which worker processed it
    retry_count INT DEFAULT 0,

    -- Result
    result JSONB, -- Agent output
    error_message TEXT,
    error_type VARCHAR(100),

    -- Timing
    queued_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    execution_time_ms INT,

    -- Credentials used (for auditing, NOT the actual tokens)
    credential_id UUID REFERENCES user_credentials(credential_id) ON DELETE SET NULL,
    llm_credential_id UUID REFERENCES user_credentials(credential_id) ON DELETE SET NULL
);

CREATE INDEX idx_tasks_tenant_user ON agent_tasks(tenant_id, user_id);
CREATE INDEX idx_tasks_session ON agent_tasks(session_id, queued_at DESC);
CREATE INDEX idx_tasks_status ON agent_tasks(status, queued_at) WHERE status IN ('queued', 'running');

-- 6. User Credentials (already defined in section 4.2)
-- See earlier schema

-- 7. Audit Log (immutable log for compliance)
CREATE TABLE audit_log (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,

    -- Event details
    action VARCHAR(100) NOT NULL, -- 'credential.created', 'session.started', 'task.executed'
    resource_type VARCHAR(100), -- 'user_credential', 'chat_session', 'agent_task'
    resource_id UUID,

    -- Context
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,

    -- Timestamp (immutable)
    created_at TIMESTAMP DEFAULT NOW() NOT NULL
);

CREATE INDEX idx_audit_tenant_time ON audit_log(tenant_id, created_at DESC);
CREATE INDEX idx_audit_user_time ON audit_log(user_id, created_at DESC) WHERE user_id IS NOT NULL;
CREATE INDEX idx_audit_action ON audit_log(action, created_at DESC);

-- Prevent updates/deletes on audit log
CREATE RULE audit_log_no_update AS ON UPDATE TO audit_log DO INSTEAD NOTHING;
CREATE RULE audit_log_no_delete AS ON DELETE TO audit_log DO INSTEAD NOTHING;
```

### 7.2 Row-Level Security (RLS)

```sql
-- Enable RLS on all tenant-scoped tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_credentials ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see data from their own tenant
CREATE POLICY tenant_isolation_users ON users
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_sessions ON chat_sessions
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_messages ON messages
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_tasks ON agent_tasks
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_credentials ON user_credentials
    FOR ALL
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

-- Usage in application: Set tenant context for each request
-- SET LOCAL app.current_tenant_id = '123e4567-e89b-12d3-a456-426614174000';
```

### 7.3 Database Middleware (Tenant Context)

```python
# infrastructure/database.py

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from contextvars import ContextVar
import uuid

# Thread-safe context variable for tenant_id
current_tenant_id: ContextVar[uuid.UUID] = ContextVar('current_tenant_id')

class TenantDatabase:
    """Database client with automatic tenant isolation"""

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_size=20, max_overflow=40)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Set tenant context on connection
        @event.listens_for(self.engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            # This runs for every new connection from pool
            pass

    async def get_session(self, tenant_id: uuid.UUID):
        """Get database session with tenant context set"""
        db = self.SessionLocal()

        try:
            # Set tenant context for RLS
            db.execute(
                "SET LOCAL app.current_tenant_id = :tenant_id",
                {'tenant_id': str(tenant_id)}
            )

            # Store in context var for application use
            current_tenant_id.set(tenant_id)

            yield db
        finally:
            db.close()

# FastAPI dependency
async def get_db(
    current_user: User = Depends(get_current_user)
) -> AsyncGenerator[Session, None]:
    """Dependency that provides database session with tenant isolation"""
    async for db in database.get_session(current_user.tenant_id):
        yield db
```

### 7.4 Data Partitioning Strategy

```sql
-- Partition large tables by tenant_id for performance

-- Example: Partition messages table (high volume)
CREATE TABLE messages_partitioned (
    LIKE messages INCLUDING ALL
) PARTITION BY HASH (tenant_id);

-- Create 16 partitions (adjust based on scale)
CREATE TABLE messages_partition_0 PARTITION OF messages_partitioned
    FOR VALUES WITH (MODULUS 16, REMAINDER 0);

CREATE TABLE messages_partition_1 PARTITION OF messages_partitioned
    FOR VALUES WITH (MODULUS 16, REMAINDER 1);
-- ... up to partition_15

-- Indexes on each partition
CREATE INDEX idx_messages_p0_session ON messages_partition_0(session_id, sequence_number);
-- ... for each partition
```

---

## 8. MCP Connection Management

### 8.1 The Core Challenge

**Current**: Spawn subprocess per session
```bash
npx @modelcontextprotocol/server-slack  # 1 process per user
```

**Problem at Scale**:
- 1,000 users × 4 MCP agents = 4,000 Node.js processes
- ~200MB RAM each = **800GB total RAM**
- Cold start: 1-2s per process
- No connection reuse

### 8.2 Solution: MCP HTTP Adapter

**Build a wrapper that converts stdio MCP to HTTP/SSE**

```
┌─────────────────────────────────────────────┐
│  MCP HTTP Adapter (long-running service)    │
│  - Single MCP subprocess per integration    │
│  - HTTP API for tool invocation             │
│  - Connection pooling per credential set    │
│  - Horizontal scaling                       │
└─────────────────────────────────────────────┘
         │
         ├─ MCP subprocess: Slack
         ├─ MCP subprocess: GitHub
         ├─ MCP subprocess: Jira
         └─ MCP subprocess: Notion
```

#### Implementation

```python
# New service: mcp_adapter/server.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import hashlib
from typing import Dict, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = FastAPI(title="MCP HTTP Adapter")

class MCPConnectionPool:
    """Pool MCP connections by (integration_type, credentials_hash)"""

    def __init__(self):
        self.connections: Dict[str, ClientSession] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    def _get_pool_key(self, integration_type: str, credentials: dict) -> str:
        """Generate unique key for credential set"""
        cred_hash = hashlib.sha256(
            json.dumps(credentials, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"{integration_type}:{cred_hash}"

    async def get_connection(
        self,
        integration_type: str,
        credentials: dict,
        max_idle_time: int = 300  # 5 minutes
    ) -> ClientSession:
        """Get or create MCP connection"""

        pool_key = self._get_pool_key(integration_type, credentials)

        # Ensure lock exists
        if pool_key not in self.locks:
            self.locks[pool_key] = asyncio.Lock()

        async with self.locks[pool_key]:
            # Check if connection exists and is healthy
            if pool_key in self.connections:
                session = self.connections[pool_key]

                # Health check
                try:
                    await asyncio.wait_for(
                        session.list_tools(),
                        timeout=2.0
                    )
                    return session  # Reuse existing
                except:
                    # Connection dead, remove it
                    del self.connections[pool_key]

            # Create new connection
            session = await self._create_connection(integration_type, credentials)
            self.connections[pool_key] = session

            # Schedule cleanup after idle time
            asyncio.create_task(
                self._cleanup_idle_connection(pool_key, max_idle_time)
            )

            return session

    async def _create_connection(
        self,
        integration_type: str,
        credentials: dict
    ) -> ClientSession:
        """Create new MCP subprocess connection"""

        # Map integration type to MCP server package
        server_packages = {
            'slack': '@modelcontextprotocol/server-slack',
            'github': '@modelcontextprotocol/server-github',
            'jira': '@modelcontextprotocol/server-jira',
            'notion': '@modelcontextprotocol/server-notion'
        }

        if integration_type not in server_packages:
            raise ValueError(f"Unknown integration: {integration_type}")

        package = server_packages[integration_type]

        # Prepare environment with credentials
        env = {**os.environ}

        if integration_type == 'slack':
            env['SLACK_BOT_TOKEN'] = credentials['SLACK_BOT_TOKEN']
            env['SLACK_TEAM_ID'] = credentials['SLACK_TEAM_ID']
        elif integration_type == 'github':
            env['GITHUB_PERSONAL_ACCESS_TOKEN'] = credentials['GITHUB_PERSONAL_ACCESS_TOKEN']
        # ... etc

        # Spawn MCP subprocess
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", package],
            env=env
        )

        stdio_transport = await stdio_client(server_params)
        stdio, write = stdio_transport

        session = ClientSession(stdio, write)
        await session.initialize()

        return session

    async def _cleanup_idle_connection(self, pool_key: str, idle_time: int):
        """Remove connection after idle period"""
        await asyncio.sleep(idle_time)

        async with self.locks[pool_key]:
            if pool_key in self.connections:
                # Close connection
                # TODO: Add proper cleanup method
                del self.connections[pool_key]

# Global pool
connection_pool = MCPConnectionPool()

# API Endpoints

class ToolInvocationRequest(BaseModel):
    integration_type: str
    credentials: Dict[str, str]  # Decrypted credentials
    tool_name: str
    arguments: Dict

@app.post("/invoke-tool")
async def invoke_tool(request: ToolInvocationRequest):
    """
    Invoke MCP tool via HTTP

    This endpoint:
    1. Gets/creates pooled MCP connection for credentials
    2. Invokes tool
    3. Returns result
    """
    try:
        # Get pooled connection
        session = await connection_pool.get_connection(
            integration_type=request.integration_type,
            credentials=request.credentials
        )

        # Invoke tool
        result = await session.call_tool(
            name=request.tool_name,
            arguments=request.arguments
        )

        return {
            'success': True,
            'result': result.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-tools/{integration_type}")
async def list_tools(
    integration_type: str,
    # Credentials passed via headers (for pooling)
):
    """List available tools for integration"""
    # Simplified - in production, handle credentials securely
    pass

@app.get("/health")
async def health():
    """Health check"""
    return {
        'status': 'healthy',
        'active_connections': len(connection_pool.connections)
    }
```

### 8.3 Agent Executor Integration

```python
# Modified: workers/agent_executor.py

import httpx

class AgentExecutor:
    """Executes agent tasks using MCP HTTP Adapter"""

    def __init__(self, mcp_adapter_url: str = "http://mcp-adapter-service:8000"):
        self.mcp_adapter_url = mcp_adapter_url
        self.http_client = httpx.AsyncClient(timeout=120.0)

    async def execute_agent(
        self,
        agent_name: str,
        instruction: str,
        credentials: dict,
        llm_credentials: dict,
        context: Optional[dict] = None
    ) -> dict:
        """Execute agent task"""

        # For MCP-based agents, use HTTP adapter
        if agent_name in ['slack', 'github', 'jira', 'notion']:
            return await self._execute_mcp_agent(
                agent_name, instruction, credentials, llm_credentials, context
            )
        else:
            # For non-MCP agents (browser, scraper, etc.), execute directly
            return await self._execute_direct_agent(
                agent_name, instruction, llm_credentials, context
            )

    async def _execute_mcp_agent(
        self,
        agent_name: str,
        instruction: str,
        credentials: dict,
        llm_credentials: dict,
        context: Optional[dict]
    ) -> dict:
        """Execute MCP agent via HTTP adapter"""

        # 1. Decompose instruction into tool calls (using LLM)
        tool_calls = await self._decompose_instruction(
            agent_name, instruction, llm_credentials
        )

        # 2. Execute each tool call via MCP adapter
        results = []

        for tool_call in tool_calls:
            response = await self.http_client.post(
                f"{self.mcp_adapter_url}/invoke-tool",
                json={
                    'integration_type': agent_name,
                    'credentials': credentials,  # Decrypted
                    'tool_name': tool_call['tool'],
                    'arguments': tool_call['arguments']
                }
            )

            if response.status_code == 200:
                results.append(response.json()['result'])
            else:
                raise Exception(f"MCP tool invocation failed: {response.text}")

        # 3. Synthesize results
        final_result = await self._synthesize_results(
            results, llm_credentials
        )

        return {
            'agent': agent_name,
            'result': final_result,
            'tool_calls': len(tool_calls)
        }
```

### 8.4 Deployment Architecture

```yaml
# k8s/mcp-adapter-deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-adapter
spec:
  replicas: 10  # Scale independently of workers
  template:
    spec:
      containers:
      - name: mcp-adapter
        image: aerius/mcp-adapter:latest
        ports:
        - containerPort: 8000
        env:
        - name: MAX_CONNECTIONS_PER_POOL
          value: "100"
        - name: CONNECTION_IDLE_TIMEOUT
          value: "300"  # 5 minutes
        resources:
          requests:
            memory: "2Gi"  # Enough for ~100 MCP processes
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-adapter-service
spec:
  selector:
    app: mcp-adapter
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP  # Internal only
```

### 8.5 Connection Pooling Benefits

**Before (1,000 concurrent users)**:
- 4,000 MCP processes (4 agents × 1,000 users)
- 800GB RAM
- 1-2s cold start per user

**After (with pooling)**:
- ~100-200 MCP processes (shared across users with same credentials)
- ~20-40GB RAM
- Instant reuse (no cold start)
- 10 MCP adapter pods × 2GB = 20GB total

**Savings**: **97.5% reduction in memory usage**

---

## 9. Security Architecture

### 9.1 Security Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Network Security                      │
│  - VPC with private subnets                     │
│  - Security groups (least privilege)            │
│  - WAF (Web Application Firewall)               │
│  - DDoS protection (CloudFlare/AWS Shield)      │
└─────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────┐
│  Layer 2: API Gateway Security                   │
│  - Rate limiting (per-user, per-tenant)          │
│  - IP allowlisting (enterprise plan)             │
│  - Request validation (schema enforcement)       │
│  - CORS policies                                 │
└─────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────┐
│  Layer 3: Authentication & Authorization         │
│  - JWT tokens (RS256, short-lived)               │
│  - Refresh token rotation                        │
│  - OAuth2 integration (Google, GitHub)           │
│  - MFA (Multi-Factor Authentication)             │
│  - RBAC (Role-Based Access Control)              │
└─────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────┐
│  Layer 4: Data Security                          │
│  - Encryption at rest (AES-256)                  │
│  - Encryption in transit (TLS 1.3)               │
│  - Credential vault (AWS KMS / HashiCorp Vault)  │
│  - Database RLS (Row-Level Security)             │
│  - PII masking in logs                           │
└─────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────┐
│  Layer 5: Application Security                   │
│  - Input sanitization                            │
│  - SQL injection prevention (parameterized)      │
│  - XSS prevention (CSP headers)                  │
│  - CSRF tokens                                   │
│  - Secure session management                     │
└─────────────────────────────────────────────────┘
            │
┌───────────▼──────────────────────────────────────┐
│  Layer 6: Monitoring & Audit                     │
│  - Immutable audit logs                          │
│  - Anomaly detection (failed logins, etc.)       │
│  - SIEM integration                              │
│  - Compliance reporting (SOC 2, GDPR)            │
└─────────────────────────────────────────────────┘
```

### 9.2 JWT Authentication

```python
# auth/jwt_manager.py

from jose import JWTError, jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import secrets

class JWTManager:
    """Manages JWT token generation and validation"""

    def __init__(self):
        # Load RSA keys (private for signing, public for verification)
        # In production, load from secure storage (AWS Secrets Manager)
        with open('keys/private_key.pem', 'rb') as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )

        with open('keys/public_key.pem', 'rb') as f:
            self.public_key = serialization.load_pem_public_key(f.read())

    def create_access_token(
        self,
        user_id: str,
        tenant_id: str,
        role: str,
        expires_delta: timedelta = timedelta(minutes=15)
    ) -> str:
        """Create short-lived access token (15 min)"""

        now = datetime.utcnow()

        payload = {
            'sub': user_id,  # Subject (user ID)
            'tenant_id': tenant_id,
            'role': role,
            'iat': now,  # Issued at
            'exp': now + expires_delta,  # Expiration
            'jti': secrets.token_urlsafe(16),  # JWT ID (for revocation)
            'type': 'access'
        }

        token = jwt.encode(
            payload,
            self.private_key,
            algorithm='RS256'
        )

        return token

    def create_refresh_token(
        self,
        user_id: str,
        tenant_id: str,
        expires_delta: timedelta = timedelta(days=30)
    ) -> str:
        """Create long-lived refresh token (30 days)"""

        now = datetime.utcnow()

        payload = {
            'sub': user_id,
            'tenant_id': tenant_id,
            'iat': now,
            'exp': now + expires_delta,
            'jti': secrets.token_urlsafe(32),
            'type': 'refresh'
        }

        token = jwt.encode(payload, self.private_key, algorithm='RS256')

        # Store refresh token in database (for revocation)
        # await db.refresh_tokens.create({...})

        return token

    async def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""

        try:
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=['RS256']
            )

            # Check if token is revoked (check database)
            jti = payload.get('jti')
            if await self._is_token_revoked(jti):
                raise HTTPException(status_code=401, detail="Token revoked")

            return payload

        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    async def _is_token_revoked(self, jti: str) -> bool:
        """Check if token has been revoked"""
        # Check Redis cache (fast lookup)
        # Check database if not in cache
        pass

# FastAPI dependency
async def get_current_user(
    authorization: str = Header(...),
    jwt_manager: JWTManager = Depends(get_jwt_manager),
    db: Database = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user"""

    # Extract token from header: "Bearer <token>"
    scheme, token = authorization.split()
    if scheme.lower() != 'bearer':
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")

    # Verify token
    payload = await jwt_manager.verify_token(token)

    # Load user from database
    user = await db.users.find_unique(
        where={'user_id': payload['sub']}
    )

    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return user
```

### 9.3 Rate Limiting

```python
# middleware/rate_limiter.py

from fastapi import Request, HTTPException
import redis.asyncio as redis
from datetime import datetime, timedelta

class RateLimiter:
    """Token bucket rate limiting per user and tenant"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def check_rate_limit(
        self,
        user_id: str,
        tenant_id: str,
        endpoint: str,
        limits: dict
    ):
        """
        Check rate limits at user and tenant level

        limits = {
            'user': {'requests': 100, 'window': 60},  # 100 req/min per user
            'tenant': {'requests': 1000, 'window': 60}  # 1000 req/min per tenant
        }
        """
        # Check user-level limit
        user_key = f"ratelimit:user:{user_id}:{endpoint}"
        user_allowed = await self._check_limit(
            user_key,
            limits['user']['requests'],
            limits['user']['window']
        )

        if not user_allowed:
            raise HTTPException(
                status_code=429,
                detail="User rate limit exceeded",
                headers={'Retry-After': str(limits['user']['window'])}
            )

        # Check tenant-level limit
        tenant_key = f"ratelimit:tenant:{tenant_id}:{endpoint}"
        tenant_allowed = await self._check_limit(
            tenant_key,
            limits['tenant']['requests'],
            limits['tenant']['window']
        )

        if not tenant_allowed:
            raise HTTPException(
                status_code=429,
                detail="Tenant rate limit exceeded",
                headers={'Retry-After': str(limits['tenant']['window'])}
            )

    async def _check_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Token bucket algorithm using Redis"""

        now = datetime.utcnow().timestamp()

        # Lua script for atomic rate limiting
        lua_script = """
        local key = KEYS[1]
        local max_requests = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])

        -- Remove old entries outside window
        redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

        -- Count requests in current window
        local count = redis.call('ZCARD', key)

        if count < max_requests then
            -- Add current request
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, window)
            return 1
        else
            return 0
        end
        """

        result = await self.redis.eval(
            lua_script,
            1,  # Number of keys
            key,
            max_requests,
            window_seconds,
            now
        )

        return result == 1

# FastAPI middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests"""

    # Skip for health checks
    if request.url.path == "/health":
        return await call_next(request)

    # Extract user from JWT
    current_user = await get_current_user(request)

    # Define limits per endpoint
    limits = {
        '/api/v1/chat/sessions': {
            'user': {'requests': 10, 'window': 60},
            'tenant': {'requests': 100, 'window': 60}
        },
        '/api/v1/chat/messages': {
            'user': {'requests': 50, 'window': 60},
            'tenant': {'requests': 500, 'window': 60}
        }
    }

    endpoint_limits = limits.get(request.url.path, {
        'user': {'requests': 100, 'window': 60},
        'tenant': {'requests': 1000, 'window': 60}
    })

    # Check limits
    await rate_limiter.check_rate_limit(
        user_id=current_user.user_id,
        tenant_id=current_user.tenant_id,
        endpoint=request.url.path,
        limits=endpoint_limits
    )

    response = await call_next(request)
    return response
```

### 9.4 Credential Rotation

```python
# Background job: Validate and rotate credentials

import asyncio
from datetime import datetime, timedelta

class CredentialRotationService:
    """Periodic credential validation and rotation"""

    async def run_validation_job(self):
        """Run every 24 hours"""
        while True:
            print("Starting credential validation job...")

            # Find all active credentials
            credentials = await db.user_credentials.find_many(
                where={'is_active': True}
            )

            for cred in credentials:
                try:
                    # Decrypt
                    decrypted = await vault.decrypt_credentials(
                        cred.encrypted_credentials,
                        cred.encryption_key_id
                    )

                    # Validate
                    is_valid, error, metadata = await vault.validate_token(
                        cred.integration_type,
                        decrypted
                    )

                    # Update status
                    await db.user_credentials.update(
                        where={'credential_id': cred.credential_id},
                        data={
                            'validation_status': 'valid' if is_valid else 'invalid',
                            'validation_error': error,
                            'last_validated_at': datetime.utcnow(),
                            'token_metadata': metadata
                        }
                    )

                    # Alert user if invalid
                    if not is_valid:
                        await self._alert_user_invalid_token(cred)

                except Exception as e:
                    print(f"Validation failed for {cred.credential_id}: {e}")

            print(f"Validated {len(credentials)} credentials")

            # Sleep 24 hours
            await asyncio.sleep(86400)

    async def _alert_user_invalid_token(self, credential):
        """Send email/notification to user about invalid token"""
        # Send email via SendGrid/SES
        # Push notification via WebSocket if online
        pass
```

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-8) - 2 months

**Goal**: Core multi-tenant infrastructure

| Week | Tasks | Team |
|------|-------|------|
| 1-2 | • Database schema design<br>• PostgreSQL + Redis setup<br>• RLS policies | Backend (2) |
| 3-4 | • FastAPI web server<br>• JWT authentication<br>• User registration/login | Backend (2) + Frontend (1) |
| 5-6 | • BYOT credential management<br>• KMS encryption setup<br>• Credential vault API | Backend (2) + Security |
| 7-8 | • Session management<br>• Message API<br>• Basic chat interface | Full Stack (3) |

**Deliverable**: Working multi-tenant chat system (no agents yet)

---

### Phase 2: Kafka + Workers (Weeks 9-14) - 1.5 months

**Goal**: Async task processing

| Week | Tasks | Team |
|------|-------|------|
| 9-10 | • Kafka cluster setup (AWS MSK)<br>• Topic design + schemas<br>• Producer implementation | Backend (2) + DevOps |
| 11-12 | • Worker pods (Kubernetes)<br>• Consumer implementation<br>• Task execution framework | Backend (2) |
| 13-14 | • Result publishing<br>• WebSocket real-time updates<br>• Status tracking | Backend (2) + Frontend (1) |

**Deliverable**: End-to-end async task processing

---

### Phase 3: MCP Integration (Weeks 15-20) - 1.5 months

**Goal**: Agent functionality with BYOT

| Week | Tasks | Team |
|------|-------|------|
| 15-16 | • MCP HTTP adapter design<br>• Connection pooling<br>• Slack + GitHub adapters | Backend (2) |
| 17-18 | • Jira + Notion adapters<br>• Agent executor refactor<br>• LLM integration | Backend (2) |
| 19-20 | • Integration testing<br>• Performance optimization<br>• Bug fixes | Full Team (4) |

**Deliverable**: All 4 MCP agents working with BYOT

---

### Phase 4: Scaling + Production (Weeks 21-26) - 1.5 months

**Goal**: Production-ready system

| Week | Tasks | Team |
|------|-------|------|
| 21-22 | • Load balancing (nginx)<br>• Auto-scaling config<br>• Load testing | DevOps + Backend (2) |
| 23-24 | • Monitoring (Prometheus/Grafana)<br>• Alerting<br>• Log aggregation | DevOps + Backend |
| 25-26 | • Security audit<br>• Penetration testing<br>• Compliance (SOC 2 prep) | Security + Full Team |

**Deliverable**: Production deployment

---

### Phase 5: Polish + Launch (Weeks 27-30) - 1 month

| Week | Tasks | Team |
|------|-------|------|
| 27-28 | • UI/UX improvements<br>• Onboarding flow<br>• Documentation | Frontend (1) + PM |
| 29-30 | • Beta testing<br>• Bug fixes<br>• Launch preparation | Full Team (4) |

**Total**: **7.5 months** with 3-4 engineers

---

## 11. Cost Analysis & Trade-offs

### 11.1 Infrastructure Costs (AWS, 1,000 concurrent users)

| Service | Specs | Monthly Cost |
|---------|-------|--------------|
| **EC2 (FastAPI)** | 10 × t3.large (2 vCPU, 8GB) | $600 |
| **EC2 (Workers)** | 50 × t3.xlarge (4 vCPU, 16GB) | $6,000 |
| **EC2 (MCP Adapter)** | 10 × t3.large (2 vCPU, 8GB) | $600 |
| **RDS PostgreSQL** | db.r6g.2xlarge (8 vCPU, 64GB) + replicas | $1,500 |
| **ElastiCache Redis** | cache.r6g.large (2 vCPU, 13GB) × 2 nodes | $400 |
| **MSK (Kafka)** | kafka.m5.large × 3 brokers + storage | $900 |
| **ALB** | Load balancer + data transfer | $200 |
| **S3** | Session logs (1TB/month) | $25 |
| **AWS KMS** | Encryption keys + API calls | $50 |
| **CloudWatch** | Logs + metrics | $300 |
| **Data Transfer** | Outbound (500GB/month) | $45 |

**Total Infrastructure**: **~$10,620/month** for 1,000 concurrent users

**Per-user cost**: **$10.62/month**

### 11.2 Cost Optimization Strategies

1. **Reserved Instances**: Save 40% on EC2 (1-year commitment)
   - Reduction: $10,620 → $7,500/month

2. **Spot Instances for Workers**: Save 70% on worker nodes
   - Reduction: $6,000 → $1,800/month

3. **Auto-scaling Aggressiveness**: Scale down during off-peak
   - Reduction: 30% savings = $2,250/month

**Optimized Total**: **~$5,000-6,000/month** for 1,000 users

### 11.3 Trade-offs

| Decision | Pro | Con |
|----------|-----|-----|
| **Kafka vs SQS** | Higher throughput, ordering guarantees | More complex, higher cost |
| **PostgreSQL vs DynamoDB** | Relational model, RLS, ACID | Scaling limits beyond 10M rows |
| **MCP HTTP Adapter vs Direct** | 95% cost reduction, shared connections | Added complexity, single point of failure |
| **Monorepo vs Microservices** | Simpler deployment, shared code | Harder to scale teams |
| **Self-hosted vs Managed** | Lower cost at scale | Higher operational burden |

---

## 12. Migration Strategy

### 12.1 Backward Compatibility

**Keep CLI version alongside web service**:

```
┌──────────────────────────────────────────┐
│  Current CLI (unchanged)                 │
│  - Single-user mode                      │
│  - Local file storage                    │
│  - Environment variables                 │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  New Web Service                         │
│  - Multi-tenant                          │
│  - Database storage                      │
│  - BYOT credentials                      │
└──────────────────────────────────────────┘
```

### 12.2 Gradual Rollout

**Phase 1**: Internal beta (company employees)
- Validate core functionality
- Fix critical bugs
- Gather feedback

**Phase 2**: Private beta (50 users)
- Onboard friendly customers
- Monitor performance
- Iterate on UX

**Phase 3**: Public beta (500 users)
- Open signups (waitlist)
- Scale testing
- Marketing campaigns

**Phase 4**: General availability
- Remove waitlist
- Full production launch

---

## Conclusion

This is a **massive architectural transformation** that will take **6-9 months** with a dedicated team. The key challenges are:

1. **MCP connection pooling** (hardest technical problem)
2. **Multi-tenant data isolation** (security critical)
3. **BYOT credential management** (UX + security)
4. **Kafka-based scaling** (operational complexity)

**My honest assessment**:
- ✅ **Feasible**: All components are well-understood technologies
- ⚠️ **Complex**: Requires senior engineers with distributed systems experience
- ⚠️ **Costly**: ~$5-10K/month infrastructure + $500K+ engineering costs
- ✅ **Scalable**: Can support 10,000+ concurrent users with this architecture

**Recommendation**: Start with **Phase 1-2** (foundation + Kafka) and validate market demand before investing in full MCP adapter infrastructure. You can start with simpler subprocess-per-request model and optimize later once you have paying customers.

Would you like me to dive deeper into any specific component?
