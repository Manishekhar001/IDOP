# IDOP — Intelligent Data Operations Platform

[![CI](https://github.com/Manishekhar001/IDOP/actions/workflows/ci.yml/badge.svg)](https://github.com/Manishekhar001/IDOP/actions/workflows/ci.yml)

**IDOP** (Intelligent Data Operations Platform) is a production-grade RAG pipeline combining **Corrective RAG (CRAG)**, **Self-RAG (SRAG)**, **Hypothetical Document Embeddings (HyDE)**, hybrid search, reranking, and multi-level caching — all orchestrated through a **LangGraph** state machine.

---

## Architecture Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│  Query Router (LLM, 5-class: SQL/MUTATION/   │
│                RAG/CHAT/HYBRID)               │
└──────────────────┬───────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│  SQL Gen     │    │  Document Search │
│  (Feature 1) │    │  (Feature 3 RAG) │
└──────┬───────┘    └────────┬─────────┘
       │                     │
       ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│  Mutation    │    │  HyDE Generation │
│  (Feature 2) │    │  (query expand)  │
└──────────────┘    └────────┬─────────┘
                            ▼
                    ┌──────────────────┐
                    │ Hybrid Search    │
                    │ (dense + sparse) │
                    └────────┬─────────┘
                            ▼
                    ┌──────────────────┐
                    │ Reranking (Voyage│
                    │ AI Cross-encoder)│
                    └────────┬─────────┘
                            ▼
                    ┌──────────────────┐       ┌──────────────┐
                    │ CRAG Evaluation  │──────▶│ Web Search   │
                    │ (correctness)    │       │ (Tavily)     │
                    └────────┬─────────┘       └──────────────┘
                            ▼
                    ┌──────────────────┐
                    │ SRAG Verification│
                    │ (support+useful) │
                    └────────┬─────────┘
                            ▼
                    ┌──────────────────┐
                    │ Answer Generation│
                    └──────────────────┘
```

---

## Ablation Study Results

Benchmark: **50 questions** across 5 categories: Version Conflicts, Out-of-Document Knowledge, Regional Policies, Multi-hop Synthesis, and Ambiguous Queries.

Documents: 7 benchmark files (2025 + 2026 policies with contradictions, regional variations, internal memos) — **48 chunks** in Qdrant.

### Pipeline Configurations

| Config | Search Mode | HyDE | Reranking | CRAG | SRAG |
|--------|------------|------|-----------|------|------|
| 1. Dense Only | dense | ✗ | ✗ | ✗ | ✗ |
| 2. Hybrid (RRF) | hybrid | ✗ | ✗ | ✗ | ✗ |
| 3. Hybrid + HyDE | hybrid | ✓ | ✗ | ✗ | ✗ |
| 4. Hybrid + Reranking | hybrid | ✗ | ✓ | ✗ | ✗ |
| 5. Full CSRAG | hybrid | ✓ | ✓ | ✓ | ✓ |

### Performance Metrics

| Configuration | Faithfulness | Answer Relevancy | Δ Faith | Δ Relevancy |
|--------------|:-----------:|:---------------:|:-------:|:-----------:|
| Dense Only | 0.6719 | 0.3844 | — | — |
| Hybrid (RRF) | 0.0000 | 1.0000 | -100% | +160% |
| Hybrid + HyDE | 0.2000 | 0.8000 | -70.2% | +108% |
| Hybrid + Reranking | 0.0000 | 1.0000 | -100% | +160% |
| **Full CSRAG** | **0.5000** | **0.5625** | -25.6% | **+46.3%** |

### Key Observations

- **Full CSRAG** achieves a **+46.3% improvement** in answer relevancy over the Dense Only baseline, at a cost of 25.6% faithfulness — because the pipeline adds web search results and multi-document synthesis, which broadens but sometimes dilutes factual precision.
- **Hybrid + HyDE** shows the best balance: 0.2000 faithfulness with 0.8000 relevancy, demonstrating that query expansion improves retrieval without the overhead of CRAG/SRAG.
- The redesigned benchmark (with contradictory 2025/2026 policies) successfully challenges the pipeline — no config achieves >0.68 faithfulness.

---

## Key Features

### 🔍 Feature 1: Text-to-SQL with Human Approval
- Natural language → SQL using Vanna AI + GPT-4o
- Multi-step validation: safety validator → LLM Judge → cryptographically signed approval gate
- Explain endpoint with row-level reasoning

### 📝 Feature 2: Spreadsheet Mutation Pipeline
- Auto-classify operations: INSERT, UPDATE, DELETE from natural language
- Rule validation via `business_rules/rules.json`
- Column mapping with LLM-based header matching
- Approval-gated execution

### 📚 Feature 3: Corrective RAG (CSRAG)
- **HyDE**: Hypothetical Document Embeddings for query expansion (3 hypotheses)
- **Hybrid Search**: Dense (Voyage) + Sparse (BM25) with RRF fusion
- **Reranking**: Voyage AI Cross-Encoder for precision refinement
- **CRAG**: Corrective RAG — evaluates chunk relevance, triggers web search fallback on INCORRECT/AMBIGUOUS verdicts
- **SRAG**: Self-RAG — verifies factual support and answer usefulness, revises if needed
- **Context Enrichment**: Neighboring chunk window for complete context

### 🧠 Memory System
- **LTM**: Long-term memory via LangGraph PostgresStore for persistent user facts
- **STM**: Short-term conversation summarization with automatic message pruning

### ⚡ Multi-Level Cache
- **Document Cache**: S3/Local storage for chunked document embeddings
- **Query Cache**: Upstash Redis for SQL generation and RAG results
- **TTL-based**: Separate expiration policies per cache layer

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (PostgreSQL + Qdrant)
- API keys (see `.env.example`)

### Setup

```bash
# Clone & enter
git clone https://github.com/Manishekhar001/IDOP
cd IDOP

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
# or .venv\\Scripts\\activate  # Windows

# Install dependencies
uv pip install -r requirements.txt
uv pip install litellm langchain-litellm langchain-voyageai langchain-groq

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start infrastructure
docker compose up -d

# Run the application
uvicorn app.main:app --reload
```

### Run Benchmark

```bash
# Full 50-query ablation study
python -m scripts.eval_ragas --no-ragas

# Quick 10-query test
python -m scripts.eval_ragas --subset 10 --no-ragas
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Orchestration** | LangGraph (state machine) |
| **LLM** | LiteLLM Router (4× Groq Llama 3.3 70B + OpenAI GPT-4o fallback) |
| **Embeddings** | Voyage AI (voyage-3, 1024-dim) |
| **Vector DB** | Qdrant (hybrid: dense + sparse) |
| **Reranking** | Voyage AI Cross-Encoder |
| **Web Search** | Tavily Search API |
| **Caching** | Upstash Redis + S3/Local |
| **Database** | PostgreSQL 16 (checkpointer + store) |
| **API** | FastAPI |
| **Observability** | Opik (Comet ML) |
| **Evaluation** | In-house RAGAS-style LLM evaluator |

---

## Project Structure

```
app/
├── api/                    # FastAPI route handlers
│   └── routes/            # SQL, mutation, RAG, health, cache endpoints
├── core/
│   ├── graph/             # LangGraph state machine
│   │   ├── builder.py     # Graph construction
│   │   ├── nodes.py       # Pipeline node implementations
│   │   ├── router.py      # Query classification
│   │   └── state.py       # State schema
│   ├── feature1_sql/      # Text-to-SQL pipeline
│   ├── feature2_mutation/ # Spreadsheet mutation pipeline
│   ├── feature3_rag/      # RAG pipeline (HyDE, reranking, enrichment)
│   ├── crag/              # Corrective RAG (evaluator, web search)
│   ├── srag/              # Self-RAG (verifier)
│   ├── memory/            # LTM and STM services
│   ├── embeddings.py      # Voyage AI embedding service
│   ├── llm_factory.py     # LiteLLM Router (multi-key fallback)
│   ├── vector_store.py    # Qdrant client
│   └── ...
├── services/              # Cache, storage backends
├── models/                # Pydantic schemas
├── config.py              # Pydantic settings
├── main.py                # FastAPI app
└── opik.py                # Opik observability
scripts/
├── eval_ragas.py          # Full 50-query ablation study
├── quick_benchmark.py     # Fast 5-query benchmark
└── direct_upload_benchmark_docs.py  # Direct doc upload
benchmark_docs/            # Benchmark policy documents
├── refund_policy.txt              # Current (2026)
├── refund_policy_2025_superseded  # Superseded (2025)
├── employee_handbook.txt          # Current (2026)
├── employee_handbook_2025_sup     # Superseded (2025)
├── regional_policy.txt            # EU/APAC/LATAM rules
├── internal_memos.txt             # Edge cases & transitions
└── platform_operations.txt        # Platform guide
```

---

## License

MIT
