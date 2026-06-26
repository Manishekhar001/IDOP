# 14-deployment: Production Architecture & Compose

This document details the production deployment design, container orchestrations, environment configurations, and continuous deployment pipelines utilized to host IDOP.

---

## Overview

Deploying an agentic platform requiring long-running asynchronous execution states, human-in-the-loop approval delays, and persistent memory caching rules out purely serverless (e.g., AWS Lambda) frameworks due to execution window constraints.

IDOP is deployed inside an optimized, containerized environment using **Docker Compose** on virtual servers (e.g., AWS EC2 or comparable VPS models), connecting to managed enterprise database and search engines to optimize scalability, resilience, and data isolation.

```mermaid
graph TD
    %% Styling Definitions
    classDef client fill:#d4e157,stroke:#9e9d24,stroke-width:2px,color:#000;
    classDef ec2 fill:#eceff1,stroke:#607d8b,stroke-width:1.5px,color:#263238;
    classDef external fill:#e1f5fe,stroke:#0288d1,stroke-width:1.5px,color:#01579b;

    Client([Web Dashboard / REST Client]) -->|Port 80| Nginx[Nginx Reverse Proxy]
    
    subgraph EC2 Instance [AWS EC2 Instance / VPS]
        Nginx -->|Port 8000| FastAPI[FastAPI Container: Python 3.11]
        FastAPI -->|Port 5432| CheckpointDB[Internal PostgreSQL Container]
    end

    %% External APIs and Managed Systems
    FastAPI -->|Managed Connection| Supabase[Supabase Managed PostgreSQL: Audit Logs & Business DB]
    FastAPI -->|Hybrid Vector Ops| QdrantCloud[Qdrant Cloud Managed Vector Store]
    FastAPI -->|Caching Layer| UpstashRedis[Upstash Managed Redis]
    FastAPI -->|File Storage Cache| AWSS3[AWS S3 Document Buckets]
    
    %% API Services
    FastAPI -->|REST API calls| LiteLLMRouter[LiteLLM Router<br/>Groq Llama 3.3 70B (primary)<br/>OpenAI gpt-4o-mini (fallback)]
    FastAPI -->|REST API calls| VoyageAI[Voyage AI Reranking API]
    FastAPI -->|REST API calls| Tavily[Tavily Search API]

    class Client client;
    class Nginx,FastAPI,CheckpointDB ec2;
    class Supabase,QdrantCloud,UpstashRedis,AWSS3,OpenAI,VoyageAI,Tavily external;
```

---

## Multi-Container Architecture (Docker Compose)

The standard virtual runtime uses Docker Compose to orchestrate local infrastructure alongside managed cloud clusters:

See the actual [docker-compose.yml](../../docker-compose.yml) at the project root for the current configuration. Key features:

- **Docker networking:** Both services communicate over a shared `idop-net` network.
- **Memory limits:** Postgres limited to 256 MB; app limited to 700 MB (prevents OOM on t2.micro 1 GB instances).
- **Postgres idle timeout prevention:** `idle_in_transaction_session_timeout=0` and `statement_timeout=0` to keep LangGraph connection pools alive.
- **Health checks:** Postgres uses `pg_isready` + `SELECT 1`; app uses `GET /health` endpoint.
- **Start period:** App has 60s `start_period` to accommodate LangGraph graph compilation.

---

## Production Environment Variables Configuration

Production configurations must be maintained as secrets. A templates file (`.env.example`) is committed to git, while the actual runtime uses a heavily guarded `.env` file:

| Environment Variable | Required | Production Value Mapping |
| :--- | :--- | :--- |
| **ENV_STATE** | Yes | `production` |
| **OPENAI_API_KEY** | Yes | Production Enterprise OpenAI Key |
| **NOMIC_API_KEY** | Yes | Nomic API key for default embeddings |
| **VOYAGE_API_KEY** | Yes | Voyage API key for reranking |
| **TAVILY_API_KEY** | Yes | Tavily key for web fallback checks |
| **QDRANT_URL** / **API_KEY** | Yes | Managed Qdrant Cloud Cluster details |
| **UPSTASH_REDIS_URL** / **TOKEN**| Yes | Distributed Upstash Redis Connection details |
| **DATABASE_URL** | Yes | Connection string pointing to internal container for STM checkpointer |
| **SUPABASE_DB_URL** | Yes | Production business DB connection (audit logs, data mutation target) |
| **S3_CACHE_BUCKET** | Yes | S3 cache bucket for document chunks |

---

## The Serverless (Lambda) Anti-Pattern

During architectural design planning, serverless options like AWS Lambda were evaluated and rejected for three key reasons:

> [!WARNING]
> 1. **Approval Gates Lifetime Limits**: The NL-to-SQL and Document-Driven Mutation pipelines require human approvals. A transaction might stay pending in the `pending_queries` memory cache for hours awaiting verification. Serverless execution environments are ephemeral and will lose state between invocations.
> 2. **Execution Timeout Limits**: CSRAG executes recursive self-correction and verification loops (generating, validating, rewriting, and re-routing). Under heavy volume, these complex graphs can run for over a minute, brushing up against the execution ceilings of API Gateways.
> 3. **Persistent Checkpoint Pooling**: Establishing connection pools from AWS Lambda to PostgreSQL databases on every request introduces massive latency penalties.

---

## CI/CD Pipeline Workflow

The build, test, and release cycle is split into two specialized, automated GitHub Actions workflows, mirroring the decoupled layout of high-governance enterprise projects:

### 1. Continuous Integration (`ci.yml`)
Triggered on any `push` or `pull_request` to the `main` branch to validate code health without deploying. See [ci.yml](../../.github/workflows/ci.yml) for the current configuration.

### 2. Continuous Deployment (`cd.yml`)
Triggered strictly on `push` to the `main` branch to orchestrate ECR publication and EC2 VM provisioning. See [cd.yml](../../.github/workflows/cd.yml) for the current configuration. The CD pipeline:
1. Builds and pushes multi-tagged Docker image to ECR
2. SSH's into EC2, writes `.env` from secrets, tags current image as `stable` for rollback
3. Pulls new image and runs `docker compose up -d`
4. Performs health check with 5 retries
5. Auto-rollbacks to `stable` tag on health check failure

---

## Related Workflows

*   [01-system-architecture](./01-system-architecture.md) - The structural component mapping details.
*   [11-memory-system](./11-memory-system.md) - Rationale for local PostgreSQL checkpoints.
*   [13-service-initialization](./13-service-initialization.md) - Lifespan checks running inside the container.
