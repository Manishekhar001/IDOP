# IDOP: Intelligent Data Operations Platform

An enterprise-grade, high-governance data orchestration platform that enables analysts to securely extract answers, perform transactional database mutations, and query unstructured knowledge repositories without writing SQL.

---

## 🏗️ Architecture Overview

IDOP acts as a robust gateway between enterprise users and database systems. It encapsulates advanced LLM-driven reasoning, memory, vector retrieval, and business constraint evaluation into a cohesive, secure platform compiled with **FastAPI** and **LangGraph**.

```mermaid
graph TD
    %% Styling Definitions
    classDef startEnd fill:#d4e157,stroke:#9e9d24,stroke-width:2px,color:#000;
    classDef gateway fill:#eceff1,stroke:#607d8b,stroke-width:1.5px,color:#263238;
    classDef router fill:#d1c4e9,stroke:#5e35b1,stroke-width:2px,color:#311b92;
    classDef engine fill:#e3f2fd,stroke:#1565c0,stroke-width:1.5px,color:#0d47a1;
    classDef database fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20;

    User([User Prompt / Document]) -->|HTTP Port 8000| API[FastAPI Gateway]
    
    API -->|Routing Decision| Router{5-Class Router}
    
    Router -->|SQL| F1[Feature 1: NL-to-SQL Vanna 2.0]
    Router -->|MUTATION| F2[Feature 2: Document Mutations]
    Router -->|RAG| F3[Feature 3: Advanced CSRAG Engine]
    Router -->|CHAT| F4[Direct Chat & Memory Nodes]
    Router -->|HYBRID| F5[Parallel SQL + RAG Synthesis]

    F1 --> SQLGate[Approval Gate] --> Supabase[Managed Supabase Postgres]
    F2 --> MutGate[Rules Validator & Token Gate] --> Supabase
    F3 --> CSRAG[HyDE + Qdrant Vector + Tavily + SRAG Loops] --> API
    F5 --> API

    class User,Supabase startEnd;
    class API gateway;
    class Router router;
    class F1,F2,F3,F4,F5,CSRAG engine;
    class SQLGate,MutGate database;
```


---

## 🧠 LangGraph State Machine & Graph Structure

IDOP uses **LangGraph** to coordinate its advanced agents, multi-tier caches, Corrective RAG (CRAG) mechanisms, Self-Reflective RAG (SRAG) self-correction loops, and database executors. The execution state is tracked within a typed context dictionary, and the execution flows dynamically through 12 registered nodes and 5 conditional edges.

### 1. State Graph Visualization

```mermaid
graph TD
    %% Styling Definitions
    classDef router fill:#d1c4e9,stroke:#5e35b1,stroke-width:2px,color:#311b92;
    classDef node fill:#eceff1,stroke:#607d8b,stroke-width:1.5px,color:#263238;
    classDef gate fill:#fff8e1,stroke:#f57f17,stroke-width:2px,color:#e65100;
    classDef endNode fill:#d4e157,stroke:#9e9d24,stroke-width:2px,color:#333;

    Start([Start Interaction]) --> Router[router_node: GPT-4o-mini]
    
    %% Router 5-paths
    Router -->|SQL| SQLGen[sql_gen_node: Vanna 2.0]
    Router -->|MUTATION| MutationNode[mutation_node: File Upload Parser]
    Router -->|CHAT| ChatGen[generate_direct_node]
    Router -->|RAG| LTMMem[ltm_remember_node]
    Router -->|HYBRID| HybridNode[hybrid_generation_node: Parallel SQL + RAG]
    
    %% SQL Path
    SQLGen --> SQLApproval[sql_approval_gate]
    SQLApproval --> END([End Turn])
    
    %% Mutation Path
    MutationNode --> MutApproval[mutation_approval_gate]
    MutApproval --> END
    
    %% Direct Chat Path
    ChatGen --> STMSum[stm_summarize_node]
    
    %% Hybrid Path
    HybridNode --> STMSum
    
    %% CSRAG Core Path
    LTMMem --> DecideRet{Decide Retrieval Needed?}
    DecideRet -->|No| ChatGen
    DecideRet -->|Yes| RetrieveDocs[retrieve_docs_node: Dense+Sparse]
    
    RetrieveDocs --> CRAGEval[evaluate_docs_node: CRAG Evaluator]
    
    CRAGEval -->|CORRECT| Refine[refine_context_node]
    CRAGEval -->|AMBIGUOUS| WebSearch[web_search_node: Tavily API]
    CRAGEval -->|INCORRECT| WebSearch
    
    WebSearch --> Refine
    Refine --> GenAnswer[generate_answer_node: GPT-4o]
    
    %% SRAG Loops
    GenAnswer --> SRAGSupport{verify_support_node}
    SRAGSupport -->|Partially / No Support & Try < 2| ReviseAns[revise_answer_node] --> GenAnswer
    SRAGSupport -->|Fully Supported| SRAGUse{verify_usefulness_node}
    
    SRAGUse -->|Not Useful & Try < 2| RewriteQ[rewrite_question_node] --> LTMMem
    SRAGUse -->|Useful| STMSum
    
    STMSum --> END

    class Router,DecideRet,SRAGSupport,SRAGUse router;
    class SQLGen,MutationNode,ChatGen,LTMMem,HybridNode,RetrieveDocs,CRAGEval,WebSearch,Refine,GenAnswer,ReviseAns,RewriteQ,STMSum node;
    class SQLApproval,MutApproval gate;
    class END endNode;
```

### 2. The `CSRAGState` State Shape

Every node in the state machine reads and updates a centralized `CSRAGState` thread-safe dictionary, which tracks:
* **Core Context**: Conversation messages, user identifiers, session threads, and raw questions.
* **Routing Flags**: The determined query type (`SQL`, `MUTATION`, `RAG`, `CHAT`, `HYBRID`) and retrieval decisions.
* **RAG Properties**: Custom thresholds, dense/sparse/hybrid search settings, HyDE hypothetical hypotheses, and Voyage AI reranking flags.
* **Retrieved Artifacts**: Native document chunks, Qdrant Dual-Vector payloads, CRAG evaluation verdicts (`CORRECT`, `AMBIGUOUS`, `INCORRECT`), Tavily web results, and refined text sentence strips.
* **Verification Loops**: Self-reflective correctness ratings (`fully_supported`, `partially_supported`, `no_support`), usefulness metrics (`useful`, `not_useful`), and retry counters to prevent infinite graph traversal.
* **SQL & Mutation Schemas**: Extracted rules, SQL query strings, audit results, parameterized bulk records, transacting details, and execution logs.
* **Security Tokens**: Secure cryptographic hexadecimal strings generated for the pending approval gates.

### 3. Explanation of Core Graph Nodes

IDOP's state graph registers 12 custom execution nodes:
1. **`router`**: Inspects user queries using a structured JSON schema on GPT-4o-mini to classify the intent into one of five paths: SQL, MUTATION, RAG, CHAT, or HYBRID.
2. **`sql_gen`**: Generates high-fidelity SQL queries using Vanna 2.0. If Vanna fails or encounters timeouts, a secondary direct GPT-4o schema fallback compiles the SQL.
3. **`mutation`**: Decodes bulk Excel/CSV files, extracts transacting rows, and executes structured mapping of spreadsheet columns to destination database headers.
4. **`ltm_remember`**: Dynamically retrieves long-term factual user preferences and background constraints stored inside the PostgreSQL Store.
5. **`decide_retrieval`**: Evaluates whether RAG query execution is necessary or if the long-term context contains sufficient answers to resolve the user's intent.
6. **`retrieve_docs`**: Triggers a high-performance hybrid dual-vector search (BM25 + text-embedding-3-small) fused by Reciprocal Rank Fusion (RRF) inside Qdrant Cloud.
7. **`evaluate_docs` (CRAG)**: Scores retrieved context relevance, labeling matches as CORRECT, AMBIGUOUS, or INCORRECT.
8. **`web_search`**: Rewrites queries and queries Tavily Search API to supply live web summaries if context matches are poor or missing.
9. **`refine_context`**: Triggers sentence-level refinement on combined contexts, removing irrelevant segments and formatting highly enriched context.
10. **`generate_answer`**: Incorporates source context, short-term memory, and long-term user facts to formulate high-quality responses via GPT-4o.
11. **`hybrid_gen`**: Executes both Feature 1 (NL-to-SQL generation/validation) and Feature 3 (Advanced RAG search) concurrently using parallel graph wiring, merging outputs into a consolidated query response.
12. **`stm_summarize`**: Checks chat history lengths, automatically compressing conversational lines exceeding threshold limits to update Short-Term Memory checkpointers.

### 4. Dynamic Conditional Routing Edges

The graph shifts execution flows based on five dynamic routing functions:
* **`route_after_router`**: Evaluates `state["query_type"]` and routes to `sql_gen`, `mutation`, `ltm_remember`, `hybrid_gen`, or `generate_direct`.
* **`route_after_decide`**: Decides whether to enter the retrieval sub-pipeline (`retrieve_docs`) or answer directly using stored short-term context.
* **`route_after_crag`**: Reads the Corrective RAG evaluation result. If labeled `CORRECT`, proceeds to `refine_context`. If labeled `AMBIGUOUS` or `INCORRECT`, routes to rewrite queries and trigger a live Tavily `web_search`.
* **`route_after_support` (SRAG 1)**: Reviews support metrics. If the answer is `partially_supported` or has `no_support` and the retry counter is under 2, loops back to `revise_answer`. Otherwise, proceeds to usefulness checks.
* **`route_after_usefulness` (SRAG 2)**: Assesses usefulness ratings. If deemed `not_useful` and rewrites are under 2, reformulates the user's query and routes back to the retrieval pipeline entry (`ltm_remember`). If `useful`, routes to memory saving (`stm_summarize`).

---

## 🚀 Key Features

### 1. Natural Language-to-SQL (NL-to-SQL)
- Powered by **Vanna 2.0** with vector-stored Golden Examples and database schemas.
- Implements a schema-backed **Direct GPT-4o Fallback** if the underlying agent framework encounters an execution ceiling.
- **SQLValidator Guardrails**: Programs strict parsing checks blocking destructive operations (`DROP`, `TRUNCATE`, `ALTER`, `GRANT`, `REVOKE`, `CREATE`, `REPLACE`, or explicit `COMMIT`/`ROLLBACK`).
- **Cryptographic Approval Gates**: Generates temporary approval tokens, keeping transactions pending in an in-memory session cache until human confirmation is received.

### 2. Document-Driven Mutations
- Safe bulk database mutations from Excel/CSV uploads.
- **OpClassifier & ColumnMapper**: Automatically identifies transaction types (`INSERT`, `UPDATE`, `DELETE`) and maps spreadsheet column headers (e.g. "Ph No") to target schema columns (e.g. `phone`).
- **Declarative Business Rules**: Validates row data against rules configured in [business_rules/rules.json](./business_rules/rules.json) (enforcing type constraints, numeric boundaries, regex patterns, allowed enums, and bulk quantity caps).
- **All-or-Nothing Transaction Safety**: Executes all mutations in isolated Postgres transactions, enforcing absolute database rollbacks on single-row failures.

### 3. Corrective-Self-Reflective RAG (CSRAG)
- **Hypothetical Document Embeddings (HyDE)**: Expands queries to bridge semantic vocabulary gaps.
- **Dual-Vector Hybrid Search**: Searches Qdrant Cloud using dense embeddings (text-embedding-3-small) and sparse keywords (BM25) fused with **Reciprocal Rank Fusion (RRF)**.
- **Voyage AI Reranking**: Places top-ranked documents at the front of context windows.
- **Corrective RAG (CRAG)**: Evaluates chunks (`CORRECT`, `AMBIGUOUS`, `INCORRECT`) and triggers live Tavily web search lookups on irrelevant contexts.
- **Self-Reflective RAG (SRAG)**: Uses multi-stage feedback loops verifying **evidence support** (revising generation on failures) and **answer usefulness** (reformulating queries).
- **Context Enrichment**: Neighbored chunk windowing retrieves adjacent chunks (`i-1`, `i+1`) for complete text paragraphs.

### 4. Enterprise Memory & Four-Tier Cache
- **Dual Memory**: Long-Term Memory (LTM Postgres Store for user facts) and Short-Term Memory (STM checkpointers with automated conversation summaries).
- **Redis Cache & Fallback**: Four namespaces (`embedding`, `rag`, `sql_gen`, `sql_result`) hosted on Upstash Redis with a seamless thread-safe **LRU Local Memory Fallback** when Redis is offline.

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
│   │   ├── schemas.py           # Pyright-Strict Schemas
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
│   └── logging_config.py        # Intercept Logger Layouts
├── business_rules/
│   └── rules.json               # Declarative Mutation Rules
├── tests/
│   ├── conftest.py              # Globally Patched Offline Fixtures
│   ├── test_caching.py          # Cache Tiers & Fallback tests
│   ├── test_features.py         # SQL Guards, Tokens, Rules Validator
│   ├── test_graph.py            # LangGraph Wiring tests
│   ├── test_router.py           # 5-Path Router tests
│   └── test_storage_backends.py # Local and S3 (moto) Storage tests
├── docs/design/                # Comprehensive Architectural Manuals
│   ├── 00-index.md              # Documentation Directory Map
│   └── 01-16-workflows.md       # Detailed Subsystem Documents
├── Dockerfile                   # Container Configurations
├── docker-compose.yml           # Local Multi-Container Services
├── requirements.txt             # Project Dependencies
└── README.md                    # Core Operation Manual
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.11 or 3.12 installed locally

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

# Qdrant Vector Cloud
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-cluster-api-key

# Managed Redis Caching
UPSTASH_REDIS_URL=your-redis-url
UPSTASH_REDIS_TOKEN=your-redis-token

# SQLite/PostgreSQL Session Store
DATABASE_URL=postgresql+psycopg://postgres:secure_passwd@localhost:5432/idop_memories
SUPABASE_DB_URL=postgresql+psycopg://postgres:supabase_passwd@supabase-host:5432/postgres
```

### Step 2: Set Up Local Services
Launch the internal PostgreSQL container (used for persistent LangGraph STM checkpoints and LTM facts storage):
```bash
docker-compose up -d
```

### Step 3: Install Dependencies
Create a Python virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/Scripts/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Running the Platform

To start the FastAPI server with hot-reloads enabled:
```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```
The application will launch on `http://localhost:8000`. You can inspect the Swagger interface at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🧪 Running Automated Tests

IDOP includes an isolated automated unit test suite with **147 test cases** verifying all cached nodes, routing classifiers, features, and database transactions offline without requiring real API keys or vector databases:

Run the entire test suite:
```bash
.venv\Scripts\python.exe -m pytest
```

---

## 📖 Architectural Manuals Directory

A comprehensive library of 16 highly structured workflow manuals is compiled under the `docs/design/` directory for detailed subsystem inspections:

*   **[01-System Architecture](./docs/design/01-system-architecture.md)**: Physical components map, FastAPI routing, Voyage/Qdrant/OpenAI stack.
*   **[02-Unified Query Flow](./docs/design/02-unified-query-flow.md)**: 5-path router structured classification, schemas, and availability.
*   **[03-Document Upload Pipeline](./docs/design/03-document-upload-pipeline.md)**: Chunking, dual-vector generation, and caching.
*   **[04-Feature 1: SQL Execution](./docs/design/04-feature1-sql-execution.md)**: Text2SQL generate, validate, audit judge, and token approval gates.
*   **[05-Feature 2: Mutation Pipeline](./docs/design/05-feature2-mutation-pipeline.md)**: Parsing, business rules.json mapping, safety judges, and Postgres transactions.
*   **[06-Feature 3: RAG Pipeline](./docs/design/06-feature3-rag-pipeline.md)**: Complete CSRAG workflow (LTM, HyDE, RRF, Rerank, CRAG Tavily web search, SRAG verifier support/usefulness).
*   **[07-LangGraph State Machine](./docs/design/07-langgraph-state-machine.md)**: LangGraph compiler, complete `CSRAGState` fields, active routing functions, checkpointers.
*   **[08-Hybrid Search Mechanics](./docs/design/08-hybrid-search.md)**: Dual-vector dense + sparse search mathematical modeling and RRF rank fusion formula.
*   **[09-CRAG Pipeline](./docs/design/09-crag-pipeline.md)**: Corrective RAG scoring thresholds and Tavily Web Search loops.
*   **[10-SRAG Pipeline](./docs/design/10-srag-pipeline.md)**: Self-Reflective RAG verifier support/usefulness loops and retry policies.
*   **[11-Memory System](./docs/design/11-memory-system.md)**: Short-term checkpoints (STM) and long-term PostgreSQL stores (LTM) facts extraction.
*   **[12-Multi-Level Caching](./docs/design/12-multi-level-cache.md)**: Multi-tier Upstash Redis TTL namespaces, S3 document caching, and LRU fallbacks.
*   **[13-Service Lifespan & Initialization](./docs/design/13-service-initialization.md)**: FastAPI lifecycle startup chronological ordering, graceful degradation statuses.
*   **[14-Production Deployment](./docs/design/14-deployment.md)**: Docker Compose orchestration, production secret variables mapping, CI/CD, and Serverless analysis.
*   **[15-Design Decisions & Interview Narrative](./docs/design/15-design-decisions-interview.md)**: Deep-dive model routing rationales, persistent checkpoint ACID pooling, and comprehensive stakeholder defense Q&A.
*   **[16-Step-by-Step EC2 Setup](./docs/design/16-production-deployment-guide.md)**: Detailed step-by-step virtual machine provisioning, Docker installs, SSL Certbot configurations, Nginx proxies, and log inspections.
