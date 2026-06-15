# IDOP Workflow Documentation Index

**Project:** Intelligent Data Operations Platform (IDOP)
**Version:** 0.1.1
**Stack:** FastAPI · LangGraph · LiteLLM Router · Groq Llama 3.3 70B · Qdrant · Supabase · PostgreSQL · Redis · S3 · Voyage AI · Nomic

---

## What IDOP Is

IDOP is a 5-path agentic platform with three core features and two additional processing modes that replaces the need for a data analyst for most routine data interactions:

- **Feature 1 — NL-to-SQL:** Natural language → validated SQL → human approval → execution on Supabase
- **Feature 2 — Document-Driven Mutations:** Upload Excel/CSV → AI maps columns → generates parameterised INSERT/UPDATE/DELETE → human approval → executes in a single transaction on Supabase
- **Feature 3 — Advanced RAG:** Hybrid search + CRAG + SRAG + HyDE + Reranking + Context Enrichment against uploaded knowledge-base documents
- **CHAT Mode:** Direct LLM response using STM + LTM memory (no retrieval)
- **HYBRID Mode:** Parallel SQL execution + RAG pipeline → unified synthesis

All five paths share one LangGraph state machine, one 5-class LLM semantic router, one memory system (STM + LTM), and one caching layer (Redis + S3 + Qdrant dedup).

---

## Workflow Files

| File | What It Covers |
|---|---|
| [01-system-architecture.md](./01-system-architecture.md) | Complete component map — every service, database, and external API |
| [02-unified-query-flow.md](./02-unified-query-flow.md) | 5-class LLM router: SQL / MUTATION / RAG / CHAT / HYBRID |
| [03-document-upload-pipeline.md](./03-document-upload-pipeline.md) | KB document ingest: parse → dual-vector embed → S3 cache → Qdrant |
| [04-feature1-sql-execution.md](./04-feature1-sql-execution.md) | NL-to-SQL: Vanna → LLM Judge → approval gate → audit log |
| [05-feature2-mutation-pipeline.md](./05-feature2-mutation-pipeline.md) | Document-driven mutations: parse → map → validate → approve → execute |
| [06-feature3-rag-pipeline.md](./06-feature3-rag-pipeline.md) | Full RAG pipeline: HyDE → hybrid search → rerank → CRAG → SRAG |
| [07-langgraph-state-machine.md](./07-langgraph-state-machine.md) | Complete LangGraph graph: all nodes, edges, routing functions, state |
| [08-hybrid-search.md](./08-hybrid-search.md) | Dense + sparse + RRF fusion: indexing and query-time mechanics |
| [09-crag-pipeline.md](./09-crag-pipeline.md) | CRAG: chunk scoring (CORRECT / AMBIGUOUS / INCORRECT) + web search |
| [10-srag-pipeline.md](./10-srag-pipeline.md) | SRAG: support verification + usefulness check + answer revision loop |
| [11-memory-system.md](./11-memory-system.md) | STM (AsyncPostgresSaver) + LTM (AsyncPostgresStore) |
| [12-multi-level-cache.md](./12-multi-level-cache.md) | Four Redis cache tiers + S3 document cache + Qdrant chunk dedup |
| [13-service-initialization.md](./13-service-initialization.md) | EC2 startup sequence: dependency order, health checks, graceful degradation |
| [14-deployment.md](./14-deployment.md) | EC2 + Docker Compose architecture, CI/CD, environment variables |
| [15-design-decisions-interview.md](./15-design-decisions-interview.md) | Architectural rationales, stakeholder Q&A defense, performance metrics |
| [16-production-deployment-guide.md](./16-production-deployment-guide.md) | Step-by-step AWS EC2 environment prep, SSL Nginx configs, ECR setups |

---

## Key Design Decisions (Locked)

| Decision | Choice | Reason |
|---|---|---|
| Deployment | EC2 + Docker Compose | Human approval gate can be pending indefinitely — Lambda 15-min timeout breaks this |
| LLM (generation, SQL judge, routing, CRAG, SRAG, memory) | **LiteLLM Router** (primary: Groq `llama-3.3-70b-versatile` w/ multi-key load balancing; fallback: OpenAI `gpt-4o-mini`) | High performance on complex tasks with automatic Groq-to-OpenAI failover |
| LLM (memory tasks) | **`get_memory_llm()`** — defaults to `llama-3.3-70b-versatile` via LiteLLM Router | Shared LLM instance for memory summarization and LTM extraction |
| LLM (SQL generation) | **Vanna 2.0** with OpenAI `gpt-4o-mini` (configurable via `VANNA_LLM_MODEL`) | Vanna's internal `OpenAILlmService` for robust NL-to-SQL; falls back to direct LLM SQL generation if Vanna is unavailable |
| Embeddings | Configurable: **Nomic** `nomic-embed-text-v1.5` (768-dim) or **Voyage AI** `voyage-3` (1024-dim) | `EMBEDDING_PROVIDER` env var controls the active provider; Nomic is default |
| Vector store | Qdrant (dense + sparse dual-vector) | Native BM25 + dense + RRF fusion in single collection |
| NL-to-SQL | Vanna 2.0 | Production-grade NL-to-SQL using OpenAILlmService + PostgresRunner + DemoAgentMemory; falls back to direct LLM SQL generation |
| Business data | Supabase | External, isolated from AI infrastructure |
| Memory data | PostgreSQL Docker on EC2 | Keeps LangGraph STM/LTM internal |
| Audit log | Supabase | Belongs with the data it audits |
| RAG answer cache | Post-verification only (CORRECT + fully_supported + useful) | Never cache unverified answers |
| Document cache | S3 (file-level SHA-256) | Works, already built in Text2SQL project |
| Query cache | Redis (Upstash) | SQL gen 24h, SQL result 15min, embeddings 7d |
| Reranking | Voyage AI `rerank-2.5` | Enterprise-grade cross-encoder reranking with generous free tier |
| Advanced RAG | HyDE + Hybrid Search + Reranking + Context Enrichment + CRAG + SRAG | Full Corrective Self-Reflective RAG pipeline |

---

## Data Flow Summary

```
User Query
    ↓
FastAPI Gateway (auth middleware)
    ↓
LangGraph Engine (IDOPEngine.aquery)
    ↓
ltm_remember_node → load user facts from PostgreSQL
    ↓
5-class LLM Router
    ├── SQL      → Feature 1 subgraph (Vanna → Judge → Approval → Execute)
    ├── MUTATION → Feature 2 subgraph (Parse → Map → Validate → Approve → Execute)
    ├── RAG      → Feature 3 subgraph (HyDE → Hybrid → Rerank → CRAG → SRAG)
    ├── CHAT     → Direct generation from STM + LTM checkpointers
    └── HYBRID   → Feature 1 + Feature 3 in parallel (LangGraph Send API) → merge
    ↓
stm_summarize_node → compress conversation → PostgreSQL checkpoint
    ↓
Response → client
```
