# IDOP: Intelligent Data Operations Platform

An enterprise-grade data orchestration platform that enables analysts to securely extract answers, perform transactional database mutations, and query unstructured knowledge repositories — all through natural language.

---

## 🏗️ Architecture

IDOP is a **FastAPI** + **LangGraph** gateway with three AI-driven features (NL-to-SQL, Document-Driven Mutations, Advanced CSRAG) sharing one state machine, one memory system, and one caching layer.

See [Architecture Diagram](./docs/design/01-system-architecture.md) for the full component map and service layout.

**5-Path LLM Semantic Router** classifies every query into: `SQL`, `MUTATION`, `RAG`, `CHAT`, or `HYBRID`.

The router dispatches through an **18-node LangGraph state machine** with 5 conditional edge functions (CRAG/SRAG self-correction loops). See [`docs/design/07-langgraph-state-machine.md`](./docs/design/07-langgraph-state-machine.md) for the complete graph definition.

For system architecture, component map, and service dependencies, see [`docs/design/01-system-architecture.md`](./docs/design/01-system-architecture.md).

---

## 🚀 Key Features

### 1. Natural Language-to-SQL
Vanna 2.0 → SQLValidator → LLM Judge → Cryptographic Approval Gate → Execution on Supabase.  
*See [`docs/design/04-feature1-sql-execution.md`](./docs/design/04-feature1-sql-execution.md)*

### 2. Document-Driven Mutations
Excel/CSV upload → Column mapping → Business rule validation → LLM audit → All-or-nothing Postgres transaction.  
*See [`docs/design/05-feature2-mutation-pipeline.md`](./docs/design/05-feature2-mutation-pipeline.md)*

### 3. Corrective Self-Reflective RAG (CSRAG)
HyDE → Dual-vector hybrid search (Qdrant) → Reranking (Voyage AI) → CRAG relevance evaluation → Tavily web fallback → SRAG support/usefulness verification loops.  
*See [`docs/design/06-feature3-rag-pipeline.md`](./docs/design/06-feature3-rag-pipeline.md)*

### 4. Memory System
- **LTM** (AsyncPostgresStore): Persistent user facts extracted by GPT-4o-mini
- **STM** (AsyncPostgresSaver): Conversation checkpoints with automated summarization (threshold: 6 messages)  
*See [`docs/design/11-memory-system.md`](./docs/design/11-memory-system.md)*

### 5. Multi-Level Cache
Four Redis namespaces (embedding/rag/sql_gen/sql_result) with local LRU fallback + S3 document chunk cache with SHA-256 dedup.  
*See [`docs/design/12-multi-level-cache.md`](./docs/design/12-multi-level-cache.md)*

### 6. Enterprise Security
Auto-encoding connection strings, CORS hardening, cryptographic single-use approval tokens, all-or-nothing transaction rollbacks, and Opik observability.  
*See [`docs/design/15-design-decisions-interview.md`](./docs/design/15-design-decisions-interview.md)*

---

## 📁 Repository Structure

```
├── app/
│   ├── api/
│   │   ├── routes/              # FastAPI Route Endpoints
│   │   │   ├── cache.py         # Cache Invalidation Controls
│   │   │   ├── chat.py          # Unified Router & CSRAG Chat Flow
│   │   │   ├── documents.py     # File Ingestion & Parsing
│   │   │   ├── health.py        # Lifecycle Health Checks
│   │   │   ├── memory.py        # STM & LTM Facts Extraction
│   │   │   ├── mutation.py      # Mutation Processing & Approvals
│   │   │   └── sql.py           # SQL Query Execution & Approvals
│   │   ├── schemas.py           # Pydantic Schemas
│   │   └── main.py              # Application Entry & Lifespan Context
│   ├── core/
│   │   ├── crag/                # Corrective RAG Evaluators & Web Crawls
│   │   ├── feature1_sql/        # Vanna Services, SQL Guards & Judgement
│   │   ├── feature2_mutation/   # Column Mapping, Transaction Execution
│   │   ├── feature3_rag/        # HyDE Expansion, Voyage Reranking
│   │   ├── graph/               # LangGraph State Graph & Node Wiring
│   │   ├── memory/              # Postgres STM and LTM Connectors
│   │   ├── srag/                # Answer Grounding & Reflection Loops
│   │   ├── embeddings.py        # Dense Embedding Clients
│   │   ├── sparse_vector_service.py # BM25 Hashing Services
│   │   └── vector_store.py      # Qdrant Dual-Vector Search
│   ├── services/
│   │   ├── cache_service.py     # Document Chunk Storage Routing
│   │   ├── local_storage.py     # Disk Storage Caches
│   │   ├── query_cache_service.py # Redis client & In-Memory LRU
│   │   └── s3_storage.py        # AWS S3 Storage Buckets
│   ├── config.py                # Pydantic Settings Validations
├── business_rules/
│   └── rules.json               # Declarative Mutation Rules
├── tests/
│   ├── conftest.py              # Globally Patched Offline Fixtures
│   ├── test_caching.py          # Cache Tiers & Fallback tests
│   ├── test_features.py         # SQL Guards, Tokens, Rules Validator
│   ├── test_graph.py            # LangGraph Wiring tests
│   ├── test_router.py           # 5-Path Router tests
│   └── test_storage_backends.py # Local and S3 (moto) Storage tests
├── docs/design/                 # Comprehensive Architectural Manuals
│   ├── 00-index.md              # Documentation Directory Map
│   └── 01-16-workflows.md       # Detailed Subsystem Documents
├── Dockerfile                   # Container Configurations
├── docker-compose.yml           # Local Multi-Container Services
├── requirements.txt             # Project Dependencies
└── README.md                    # This file
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.11+ installed locally

### Step 1: Environment Variables
Create a `.env` file in the root directory. You can copy the template from `.env.example`:
```bash
cp .env.example .env
```

Fill in the required parameters:
```ini
ENV_STATE=development
OPENAI_API_KEY=your-openai-api-key
VOYAGE_API_KEY=your-voyage-api-key
TAVILY_API_KEY=your-tavily-api-key
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-cluster-api-key
UPSTASH_REDIS_URL=your-redis-url
UPSTASH_REDIS_TOKEN=your-redis-token
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/postgres
SUPABASE_DB_URL=postgresql+psycopg://postgres:supabase_passwd@supabase-host:5432/postgres
```

### Step 2: Set Up Local Services
Start the internal PostgreSQL container (for LangGraph STM checkpoints and LTM facts):
```bash
docker-compose up -d
```

### Step 3: Install Dependencies
```bash
python -m venv .venv
source .venv/Scripts/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Platform

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The application starts on `http://localhost:8000`. Interactive docs at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Running Automated Tests

```bash
.venv\Scripts\python.exe -m pytest
```

---

## 📖 Design Documentation

Detailed architectural manuals for every subsystem:

| Doc | Topic |
|---|---|
| [00-index.md](./docs/design/00-index.md) | Documentation index & key design decisions table |
| [01-system-architecture.md](./docs/design/01-system-architecture.md) | Complete component map — every service, database, and external API |
| [02-unified-query-flow.md](./docs/design/02-unified-query-flow.md) | 5-class LLM semantic router: SQL / MUTATION / RAG / CHAT / HYBRID |
| [03-document-upload-pipeline.md](./docs/design/03-document-upload-pipeline.md) | KB document ingest: parse → dual-vector embed → S3 cache → Qdrant |
| [04-feature1-sql-execution.md](./docs/design/04-feature1-sql-execution.md) | NL-to-SQL: Vanna → LLM Judge → approval gate → audit log |
| [05-feature2-mutation-pipeline.md](./docs/design/05-feature2-mutation-pipeline.md) | Document-driven mutations: parse → map → validate → approve → execute |
| [06-feature3-rag-pipeline.md](./docs/design/06-feature3-rag-pipeline.md) | Full RAG pipeline: HyDE → hybrid search → rerank → CRAG → SRAG |
| [07-langgraph-state-machine.md](./docs/design/07-langgraph-state-machine.md) | Complete LangGraph graph: all 18 nodes, 5 conditional edges, state shape |
| [08-hybrid-search.md](./docs/design/08-hybrid-search.md) | Dense + sparse + RRF fusion: indexing and query-time mechanics |
| [09-crag-pipeline.md](./docs/design/09-crag-pipeline.md) | CRAG: chunk scoring (CORRECT / AMBIGUOUS / INCORRECT) + web search |
| [10-srag-pipeline.md](./docs/design/10-srag-pipeline.md) | SRAG: support verification + usefulness check + answer revision loop |
| [11-memory-system.md](./docs/design/11-memory-system.md) | STM (AsyncPostgresSaver) + LTM (AsyncPostgresStore) |
| [12-multi-level-cache.md](./docs/design/12-multi-level-cache.md) | Four Redis cache tiers + S3 document cache + Qdrant chunk dedup |
| [13-service-initialization.md](./docs/design/13-service-initialization.md) | EC2 startup sequence: dependency order, health checks, graceful degradation |
| [14-deployment.md](./docs/design/14-deployment.md) | EC2 + Docker Compose architecture, CI/CD, environment variables |
| [15-design-decisions-interview.md](./docs/design/15-design-decisions-interview.md) | Architectural rationales, stakeholder Q&A defense, performance metrics |
| [16-production-deployment-guide.md](./docs/design/16-production-deployment-guide.md) | Step-by-step AWS EC2 environment prep, SSL Nginx configs, ECR setups |
