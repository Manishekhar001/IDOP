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
    FastAPI -->|REST API calls| OpenAI[OpenAI API GPT-4o / GPT-4o-mini]
    FastAPI -->|REST API calls| VoyageAI[Voyage AI Reranking API]
    FastAPI -->|REST API calls| Tavily[Tavily Search API]

    class Client client;
    class Nginx,FastAPI,CheckpointDB ec2;
    class Supabase,QdrantCloud,UpstashRedis,AWSS3,OpenAI,VoyageAI,Tavily external;
```

---

## Multi-Container Architecture (Docker Compose)

The standard virtual runtime uses Docker Compose to orchestrate local infrastructure alongside managed cloud clusters:

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: idop-app
    restart: always
    ports:
      - "8000:8000"
    environment:
      - ENV_STATE=production
      - DATABASE_URL=postgresql://postgres:secure_passwd@checkpoint-db:5432/idop_memories
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - UPSTASH_REDIS_URL=${UPSTASH_REDIS_URL}
      - UPSTASH_REDIS_TOKEN=${UPSTASH_REDIS_TOKEN}
    depends_on:
      checkpoint-db:
        condition: service_healthy

  checkpoint-db:
    image: postgres:16-alpine
    container_name: idop-checkpoint-db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secure_passwd
      POSTGRES_DB: idop_memories
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

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
Triggered on any `push` or `pull_request` to the `main` branch to validate code health without deploying:
```mermaid
graph TD
    A[Push / Pull Request] --> B[Runner Disk Space Cleanup]
    B --> C[Setup Python 3.13 & Magic Libs]
    C --> D[Run Linter & Formatter: Ruff & Black]
    D --> E[Run Unit Tests: Pytest Offline]
    E --> F[Verify Dockerfile Syntax: Local amd64 Build]
    F --> G([CI Validation Green Success])
```

### 2. Continuous Deployment (`cd.yml`)
Triggered strictly on `push` to the `main` branch (such as a pull request merge) to orchestrate ECR publication and EC2 VM provisioning:
```mermaid
graph TD
    A[Merge / Push to main] --> B[Runner Disk Space Cleanup]
    B --> C[Configure AWS & STS Caller Identity Check]
    C --> D[Authenticate & Login to ECR]
    D --> E[Docker Buildx: Compile & Multi-tag Image]
    E --> F[Push Container tags latest & git-SHA to ECR]
    F --> G[Establish SSH Connection with EC2 Host]
    G --> H[Dynamically Write Secured .env File: chmod 600]
    H --> I[Authenticate & Login to ECR on Host]
    I --> J[Pull ECR Image & Trigger Compose Restart]
    J --> K{Multi-Iteration Health Checks Passed?}
    K -->|Yes| L([Deploy Complete & Active])
    K -->|No| M[Trigger Automated Container Rollback]
    M --> N([Exit Pipeline with Fail status])
```

---

## Related Workflows

*   [01-system-architecture](./01-system-architecture.md) - The structural component mapping details.
*   [11-memory-system](./11-memory-system.md) - Rationale for local PostgreSQL checkpoints.
*   [13-service-initialization](./13-service-initialization.md) - Lifespan checks running inside the container.
