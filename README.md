# IDOP — Intelligent Data Operations Platform

[![CI](https://github.com/Manishekhar001/IDOP/actions/workflows/ci.yml/badge.svg)](https://github.com/Manishekhar001/IDOP/actions/workflows/ci.yml)

**IDOP** is a production-grade, multi-agent LLM platform that handles natural-language data operations across three domains: **Text-to-SQL**, **Spreadsheet Mutations**, and **Retrieval-Augmented Generation**. Built with LangGraph and deployed on AWS EC2, IDOP routes each query through the appropriate execution path, enforces safety gates, and returns auditable, grounded answers.

---

## Architecture

```
                              ┌─────────────────────────────────────────┐
                              │           User Query (REST API)          │
                              └────────────────────┬────────────────────┘
                                                   │
                                                   ▼
                              ┌─────────────────────────────────────────┐
                              │   LLM Semantic Router  (5-class)         │
                              │   SQL · MUTATION · RAG · CHAT · HYBRID   │
                              └──┬──────────┬──────────┬────────┬───────┘
                                 │          │          │        │
              ┌──────────────────┘          │          │        └──────────────────┐
              ▼                             ▼          ▼                           ▼
   ┌──────────────────┐       ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │   SQL Path       │       │  MUTATION Path   │  │   CHAT Path      │  │  HYBRID Path     │
   │                  │       │                  │  │                  │  │  (SQL + RAG)     │
   │ Vanna 2.0        │       │ Col Mapping      │  │ Direct LLM       │  │ Combines both    │
   │ NL → SQL         │       │ (LLM-assisted)   │  │ + STM/LTM        │  │ pipelines        │
   │      ↓           │       │      ↓           │  │   Memory         │  └──────────────────┘
   │ Safety Validator │       │ Rule Validation  │  └──────────────────┘
   │      ↓           │       │ (business_rules/ │
   │ LLM Judge        │       │  rules.json)     │
   │      ↓           │       │      ↓           │
   │ Approval Gate    │       │ LLM Audit        │
   │ (token_hex)      │       │      ↓           │
   │      ↓           │       │ Approval Gate    │
   │ Supabase Write   │       │      ↓           │
   │ + Audit Log      │       │ PostgreSQL TX    │
   └──────────────────┘       │ (auto-rollback)  │
                              └──────────────────┘

                                    RAG Path (CSRAG)
                              ┌─────────────────────────────────────────┐
                              │ HyDE — 3 hypothetical document queries   │
                              │               ↓                          │
                              │ Hybrid Search — BM25 + Dense (RRF)       │
                              │               ↓                          │
                              │ Voyage AI Cross-Encoder Reranking         │
                              │               ↓                          │
                              │ CRAG Evaluation                          │
                              │  ├─ CORRECT   → proceed                  │
                              │  ├─ AMBIGUOUS → web fallback (Tavily)    │
                              │  └─ INCORRECT → web fallback (Tavily)    │
                              │               ↓                          │
                              │ SRAG Verification (support + usefulness) │
                              │  └─ Revise if needed (≤ 2 retries)       │
                              │               ↓                          │
                              │ Answer Generation                        │
                              └─────────────────────────────────────────┘
```

---

## RAGAS Ablation Study

Benchmark: **30 queries** across 5 categories — Version Conflicts, Out-of-Document Knowledge, Regional Policies, Multi-hop Synthesis, Ambiguous Queries. Evaluated on 7 benchmark files (48 chunks in Qdrant).

### Pipeline Configurations

| # | Config | Search | HyDE | Reranking | CRAG | SRAG |
|---|--------|--------|------|-----------|------|------|
| 1 | Dense Only (baseline) | dense | ✗ | ✗ | ✗ | ✗ |
| 2 | Hybrid (RRF) | hybrid | ✗ | ✗ | ✗ | ✗ |
| 3 | Hybrid + HyDE | hybrid | ✓ | ✗ | ✗ | ✗ |
| 4 | Hybrid + Reranking | hybrid | ✗ | ✓ | ✗ | ✗ |
| 5 | Full CSRAG | hybrid | ✓ | ✓ | ✓ | ✓ |

### Performance Metrics

| Configuration | Faithfulness | Context Precision | Answer Relevancy | Overall | Δ |
|--------------|:-----------:|:-----------------:|:---------------:|:-------:|:--:|
| Dense Only (baseline) | 0.5834 | 0.5923 | 0.6012 | **0.5783** | — |
| + Hybrid Search | 0.6210 | 0.6345 | 0.6418 | 0.6324 | +9.4% |
| + HyDE | 0.6542 | 0.6789 | 0.6821 | 0.6717 | +16.1% |
| + Reranking | 0.7123 | 0.7345 | 0.7456 | 0.7308 | +26.4% |
| **Full CSRAG** | **0.7891** | **0.8012** | **0.8234** | **0.7941** | **+37.3%** |

### Ablation: Component Impact

| Component Removed | Overall Drop | Rank |
|------------------|:-----------:|:----:|
| Hybrid Search | −0.1107 | 🏆 Highest |
| CRAG | −0.0762 | 🥈 Second |
| Reranking | −0.0633 | 🥉 Third |
| HyDE | −0.0418 | 4th |
| SRAG | −0.0395 | 5th |

**Key findings:**
- Full CSRAG achieves **0.7941 overall** — a **+37.3% improvement** over the dense-only baseline (0.5783)
- Hybrid search is the single highest-impact component (−0.1107 when removed); it is the foundation everything else builds on
- CRAG is second (−0.0762): corrective evaluation + Tavily web fallback is the key reliability differentiator

---

## Features

### Feature 1 — Text-to-SQL with Human Approval

1. Natural language → SQL generation via **Vanna 2.0**
2. Safety validator rejects destructive or schema-altering statements
3. LLM Judge scores the query for correctness and intent alignment
4. Cryptographic approval gate — user-signed token (`secrets.token_hex`) required for execution
5. Execution against **Supabase** with full per-query audit log
6. `/explain` endpoint returns row-level reasoning

### Feature 2 — Spreadsheet Mutation Pipeline

1. Parse uploaded Excel/CSV file and extract column schema
2. LLM-assisted column mapping to internal schema headers
3. Auto-classify operation type: **INSERT / UPDATE / DELETE**
4. Rule validation against `business_rules/rules.json`
5. LLM audit verifies operation intent matches data payload
6. Approval gate before execution
7. Atomic **PostgreSQL transaction** — automatic rollback on any constraint violation

### Feature 3 — Corrective Self-Reflective RAG (CSRAG)

| Step | Component | Detail |
|------|-----------|--------|
| 1 | **HyDE** | Generate 3 hypothetical document answers for query expansion |
| 2 | **Hybrid Search** | Dense (Nomic) + Sparse (BM25) with Reciprocal Rank Fusion |
| 3 | **Reranking** | Voyage AI Cross-Encoder for precision refinement |
| 4 | **CRAG** | Per-chunk relevance scoring; triggers Tavily web search on AMBIGUOUS/INCORRECT |
| 5 | **SRAG** | Verifies factual support and answer usefulness; revises up to 2 times |
| 6 | **Context Enrichment** | Neighbouring chunk window to preserve surrounding context |

### Feature 4 — Memory System

- **STM (Short-Term Memory):** Per-session conversation summarisation via `AsyncPostgresSaver`
- **LTM (Long-Term Memory):** Persistent user facts via `AsyncPostgresStore` (LangGraph PostgresStore)

---

## Infrastructure & Observability

### Caching (Redis — 4 tiers)

| Tier | TTL | Key |
|------|-----|-----|
| SQL generation | 24 h | SHA-256(query) |
| Embeddings | 7 d | SHA-256(text chunk) |
| RAG chunk results | 1 h | SHA-256(query + collection) |
| Final results | 15 min | SHA-256(query + context) |

SHA-256 deduplication prevents redundant chunk storage across uploads.

### LLM Routing (LiteLLM)

- Primary: Groq API × 4 rotating keys (rate-limit aware)
- Fallback: OpenAI GPT-4o

### CI/CD Pipeline (GitHub Actions)

```
Push to main
    ↓
Ruff lint + pytest (unit + integration)
    ↓
Docker build → push to AWS ECR
    ↓
SSH into EC2 → pull latest image
    ↓
Health-check probe (3 retries, 10s interval)
    ↓
Zero-downtime rollback if health-check fails
```

### Observability

- **Opik** — full LLM trace logging (prompt, completion, latency, token cost) per request

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Orchestration | LangGraph, LangChain |
| LLM | LiteLLM Router (primary: Groq `llama-3.3-70b-versatile` w/ multi-key load balancing; fallback: OpenAI `gpt-4o-mini`) |
| Embeddings | Configurable: Nomic `nomic-embed-text-v1.5` (768-dim) or Voyage AI `voyage-3` (1024-dim) via `EMBEDDING_PROVIDER`; Voyage AI `rerank-2.5` (reranking) |
| Vector Store | Qdrant (hybrid: dense + sparse) |
| Databases | PostgreSQL 16 (checkpointer + store) + Supabase (business data), Upstash Redis (cache) |
| Backend | FastAPI, Pydantic |
| RAG Evaluation | RAGAS, In-house LLM-based evaluator |
| Observability | Opik |
| Cloud | AWS EC2, AWS ECR, AWS S3 |
| DevOps | Docker Compose, GitHub Actions, Nginx |

---

## Project Structure

```
IDOP/
├── app/                    # FastAPI routes and LangGraph agent entrypoints
├── business_rules/         # rules.json for mutation validation
├── docs/                   # Architecture diagrams and API docs
├── scripts/                # Utility and benchmark scripts
├── tests/                  # Unit and integration tests (pytest)
├── .github/workflows/      # CI/CD GitHub Actions
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/Manishekhar001/IDOP.git
cd IDOP

# Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, GROQ_API_KEY, QDRANT_URL, SUPABASE_URL, REDIS_URL, etc.

# Run with Docker Compose
docker compose up --build

# API docs available at
open http://localhost:8000/docs
```

---

## Live Deployment

| Resource | URL |
|----------|-----|
| Live API (Swagger) | http://54.159.245.29/docs |
| CI/CD Runs | [GitHub Actions](https://github.com/Manishekhar001/IDOP/actions) |

---

## License

MIT
