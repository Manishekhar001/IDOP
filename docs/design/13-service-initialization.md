# 13-service-initialization: Lifecycle & Startup Sequence

This document describes the initialization sequence, dependency ordering, environment validations, and graceful degradation paths executed during the FastAPI application lifecycle.

---

## Overview

A complex platform incorporating dual vector databases, relational storage checkpointers, message queues, checkpointer engines, and deep external integrations must possess a highly predictable startup cycle. 

IDOP coordinates service instantiation inside a unified **FastAPI Lifespan Context Manager**. This sequence prevents race conditions (e.g., trying to compile the LangGraph state machine before checkpointer tables are instantiated in the PostgreSQL container) and enforces strict validation checks before admitting client traffic.

```mermaid
graph TD
    %% Styling Definitions
    classDef startEnd fill:#d4e157,stroke:#9e9d24,stroke-width:2px,color:#000;
    classDef process fill:#eceff1,stroke:#607d8b,stroke-width:1.5px,color:#263238;
    classDef check fill:#e0f7fa,stroke:#00acc1,stroke-width:1.5px,color:#006064;
    classDef critical fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c;

    A([FastAPI Lifecycle Start]) --> C1[Step 1: Initialize Logger & Stream Handlers]
    C1 --> D1[Step 2: Load dot-env & Validate Config]
    D1 --> D2[Step 3: One-time Qdrant stale collection deletion]
    D2 --> D3[Step 4: Create JWT users table]
    D3 --> D4[Step 5: Verify business tables]
    D4 --> D5[Step 6: Establish Qdrant Vector Client]
    
    D5 --> E7[Step 7: Connect LTM PostgreSQL AsyncStore]
    D5 --> F8[Step 8: Connect STM PostgreSQL AsyncSaver]
    
    E7 --> G9[Step 9: Instantiate & Compile Graph Engine]
    F8 --> G9
    
    G9 --> H10[Step 10: Lazy-init Redis Cache (deferred)]
    
    H10 --> I{All Critical Services OK?}
    I -->|Yes| J([API Live - Open Port 8000])
    I -->|No| K[Graceful Degradation Fallbacks or Safe Crash]
    
    K --> J

    class A,J startEnd;
    class B,C,D,E,F,G,H process;
    class I check;
    class K critical;
```

---

## Startup Sequence & Execution Steps

Upon initiation of Uvicorn, the context manager triggers the following sequential pipeline:

### Step 1: Structured Logging Setup
`setup_logging()` is fired first inside the lifespan context manager, configuring standard-library `logging` with a consistent `[timestamp] [name] [level] message` format. It sets the root logger to the configured `LOG_LEVEL`, removes duplicate handlers, and silences noisy third-party libraries (`httpx`, `httpcore`, `openai`, `qdrant_client`, `urllib3`, `groq`, `langgraph`) to `WARNING` level. All logs are written to stdout/stderr.

Source: [logger.py](../../app/utils/logger.py)

### Step 2: Configuration Loading
Configuration is loaded at module import time via `get_settings()` from `app.config`. The Pydantic `Settings` class reads environment variables from `.env`. (No explicit validation exception is raised for missing API keys — settings loading happens at import time.)

### Step 3: One-Time Qdrant Stale Collection Deletion
A one-time operation deletes the Qdrant collection to purge stale hash()-based sparse vectors (migrating to fastembed BM25). This block can be removed after the first successful deploy.

### Step 4: JWT Users Table Creation
The `idop_users` table is created (if not exists) via `create_users_table()` to support JWT authentication. This table stores user email, bcrypt-hashed password, and role.

### Step 5: Business Tables Verification
All business-related tables are verified at startup:
- `idop_approval_tokens` (via `approval_gate._ensure_table()`)
- `idop_mutation_approval_tokens` (via `mutation_approval_gate._ensure_table()`)
- `idop_audit_logs` (via `AuditLogger().ensure_table()`)
- `idop_pending_queries` and `idop_pending_mutations` (via `PendingStore._ensure_table()`)

### Step 6: Vector Store Registration
The `VectorStoreService` connects to the Qdrant instance. It checks if the primary collection `idop_documents` exists. If the collection is missing, it executes automated schema creation, applying:
*   Dense named vector payload configuration (Cosine, configurable dimensions: 768 for Nomic, 1024 for Voyage).
*   Sparse named vector payload configuration (BM25 keyword vectors).

### Step 4: Long-Term Memory (LTM) Setup
The `AsyncPostgresStore` establishes async pooling on the database URL. It runs:
```python
# Setup PostgreSQL schema structures for agent stores
async with store:
    await store.setup()
```
This checks if the tables representing agent memories are built and handles migrations automatically.

### Step 8: Short-Term Memory (STM) Checkpointer Setup
The `AsyncPostgresSaver` checkpointer initializes parallel connection pools. Note: LTM and STM initialization run **in parallel** using `asyncio.gather`, each with up to 8 retries (exponential backoff: 3s, 6s, 12s, 24s...) to handle Postgres cold-start delays on t2.micro. It compiles graph save states, ensuring the tables holding checkpoints for agent steps exist.

### Step 9: LangGraph Engine Assembly
The `CSRAGEngine` is constructed by feeding it the `VectorStoreService`, LTM `AsyncPostgresStore`, and STM `AsyncPostgresSaver`. The LangGraph compilations are executed, producing a thread-safe executable state machine with **17 nodes** and **5 conditional edge functions**.

### Step 10: Redis Cache Hook (Lazy)
The `QueryCacheService` and `CacheService` are lazily initialized via `cache_init.py` — the first call creates the singleton. This avoids startup failures if Redis/S3 are unavailable. Cache initialization errors are logged but do not block application startup.

---

## Related Workflows

*   [01-system-architecture](./01-system-architecture.md) - Learn how FastAPI coordinates these blocks.
*   [07-langgraph-state-machine](./07-langgraph-state-machine.md) - Graph compilation configuration details.
*   [12-multi-level-cache](./12-multi-level-cache.md) - Graceful degradation caching fallbacks.
*   [16-production-deployment-guide](./16-production-deployment-guide.md) - EC2 startup and health check configuration.
