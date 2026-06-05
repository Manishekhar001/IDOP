# 12-multi-level-cache: Four-Tier Caching Strategy

This workflow explains the design, key namespaces, TTL policies, and high-performance deduplication logic powering the IDOP **Four-Tier Caching System**.

---

## Overview

High-frequency enterprise applications face tight constraints regarding API latency, service cost, and rate-limiting limits. Under standard RAG pipelines, repeated user requests would trigger redundant LLM reasoning, vector embeddings, external web crawls, and database sweeps.

IDOP introduces a strict **Multi-Level Cache System** to bypass duplicate calculations. This includes a distributed `Upstash Redis` instance for low-latency key-value queries and a document-level chunk cache (via S3 or local disk storage) mapped by file SHA-256 hashes.

```mermaid
graph TD
    %% Styling Definitions
    classDef startEnd fill:#d4e157,stroke:#9e9d24,stroke-width:2px,color:#000;
    classDef hit fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;
    classDef miss fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c;
    classDef process fill:#eceff1,stroke:#607d8b,stroke-width:1.5px,color:#263238;

    A([Start: Request received]) --> B{Tier 1: Redis Embedding Cache?}
    B -->|Hit| C([Return Cached Embedding])
    B -->|Miss| D{Tier 2: Redis RAG Answer Cache?}
    
    D -->|Hit| E([Return Cached Answer])
    D -->|Miss| F{Tier 3: Redis SQL Generation Cache?}
    
    F -->|Hit| G([Return Cached SQL])
    F -->|Miss| H{Tier 4: Redis SQL Results Cache?}
    
    H -->|Hit| I([Return Cached Query Results])
    H -->|Miss| J[Execute full pipeline calculation]
    
    J --> K[Update corresponding cache namespaces]
    K --> L([Return live result])

    class A,L startEnd;
    class C,E,G,I hit;
    class J,K process;
    class B,D,F,H miss;
```

---

## The Four-Tier Cache Structure

The distributed Redis layer is structured into four distinct namespaces, each mapped to optimized Time-to-Live (TTL) policies reflecting the volatility of their underlying data:

| Cache Tier | Key Pattern | TTL | Eviction Rationale |
| :--- | :--- | :--- | :--- |
| **Tier 1: Embedding Cache** | `embedding:{sha256}` | **7 Days** (604,800s) | Embedding patterns for identical phrases are highly static. |
| **Tier 2: RAG Answer Cache** | `rag:{sha256_query}:{top_k}` | **1 Hour** (3,600s) | RAG contexts change as repositories are updated, requiring periodic re-evaluation. |
| **Tier 3: SQL Query Cache** | `sql_gen:{sha256_query}` | **24 Hours** (86,400s) | Translating English questions to SQL queries remains stable unless DB schemas alter. |
| **Tier 4: SQL Results Cache** | `sql_result:{sha256_sql}` | **15 Minutes** (900s) | Business database states are volatile. Fast updates prevent dirty read anomalies. |

---

## Document-Level Storage Caching

When a new document (PDF, CSV, Excel) is uploaded through `/upload`, the file content is instantly hashed using `SHA-256` to establish its unique `document_id`.

Before initiating parsing or dual-vector indexing:
1.  **Deduplication Check**: The system asks the active `StorageBackend` (S3 or local disk) if files under `pdf/{document_id}/chunks.json` already exist.
2.  **Point Search**: If they exist, parsing, chunking, and embedding calculations are completely bypassed. Chunks and embeddings are loaded straight from the storage cache into Qdrant or directly into the RAG engine context.

```python
# app/services/query_cache_service.py
import hashlib

def generate_cache_key(prefix: str, payload: str) -> str:
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{sha}"
```

---

## Local In-Memory Fallback (Graceful Degradation)

To preserve system resilience, the `QueryCacheService` is designed to degrade gracefully if connectivity with Upstash Redis is severed. 

If a connection timeout occurs:
1.  The exception is captured silently without breaking the active request.
2.  The engine falls back to a shared local in-memory dictionary (`self._local_cache: Dict[str, str]`).
3.  Metrics tracking marks `redis_status: "disconnected"` and updates health checks.

```python
# app/services/query_cache_service.py
class QueryCacheService:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._local_cache = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> bytes:
        if self.redis:
            try:
                val = await self.redis.get(key)
                if val:
                    self.hits += 1
                    return val
            except RedisError:
                logger.warning("Redis cache failed, falling back to local memory.")
        
        # Local Fallback lookup
        val = self._local_cache.get(key)
        if val:
            self.hits += 1
            return val
        
        self.misses += 1
        return None
```

---

## Cache Invalidation Endpoint

Administrative operations or content updates can enforce cache sweeps using the `/cache/clear` API:

*   **`/cache/clear`**: Clear all four tiers of Redis caching globally.
*   **`/cache/clear?prefix=sql_result`**: Selectively invalidate only Tier 4 SQL results to reflect sudden backend modifications.
*   **`/cache/clear?document_id=doc_hash`**: Erases a document's cache blocks from storage, forcing re-parsing and re-indexing.

---

## Related Workflows

*   [03-document-upload-pipeline](./03-document-upload-pipeline.md) - Document hashing and storage caching.
*   [06-feature3-rag-pipeline](./06-feature3-rag-pipeline.md) - RAG query caching.
*   [13-service-initialization](./13-service-initialization.md) - Lifespan setup of cache clients.
