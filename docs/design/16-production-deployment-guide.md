# 16-production-deployment-guide: Complete EC2 Deployment Guide

This manual provides a detailed, production-grade guide for system administrators to provision, configure, secure, and deploy the IDOP platform onto a clean AWS EC2 instance — including a complete reference for every GitHub Secret, environment variable, and integration key required by the CI/CD pipeline.

**Last Updated:** 2026-05-30
**Target Architecture:** AWS EC2 (Ubuntu 22.04 LTS, x86-64/AMD64) + Docker Compose + Nginx + Let's Encrypt
**Deployment Method:** GitHub Actions CI/CD (ci.yml + cd.yml)
**Audience:** Intermediate AWS users familiar with CLI, console, and GitHub Actions

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Prerequisites & Requirements](#2-prerequisites--requirements)
3. [AWS IAM & EC2 Setup](#3-aws-iam--ec2-setup)
4. [External Services Configuration](#4-external-services-configuration)
5. [GitHub Repository & Secrets Setup](#5-github-repository--secrets-setup)
6. [EC2 Server Provisioning](#6-ec2-server-provisioning)
7. [Docker & Environment Setup on EC2](#7-docker--environment-setup-on-ec2)
8. [Nginx Reverse Proxy & SSL](#8-nginx-reverse-proxy--ssl)
9. [Triggering Deployment & Verification](#9-triggering-deployment--verification)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview & Architecture

### 1.1 What Will Be Deployed

This guide walks you through deploying the **Intelligent Data Operations Platform (IDOP)** — a production-ready LangGraph multi-agent RAG system on AWS EC2 from scratch.

**Key Features:**
- 📄 **Multi-Agent RAG:** Upload documents, query them with intelligent agentic routing
- 🧠 **LangGraph State Engine:** Adaptive orchestration with corrective RAG, web search fallback
- 🗃️ **Vector + Relational Hybrid:** Qdrant vector DB + PostgreSQL checkpointing + Supabase
- ⚡ **Multi-Level Caching:** S3 document cache + Upstash Redis query cache
- 🔄 **Automatic Deployment:** Push to GitHub `main` branch → auto-deploy to EC2
- 🔒 **Production-Ready:** HTTPS via Let's Encrypt, secure .env injection, rollback protection

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│                     (Push to main branch)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                           │
│  CI (ci.yml):                                                    │
│  • Lint code with Ruff + Black                                   │
│  • Run Pytest unit tests (offline mocks)                         │
│  • Verify Docker image builds (linux/amd64)                      │
│                                                                  │
│  CD (cd.yml):                                                    │
│  • Build Docker image (linux/amd64)                              │
│  • Push to Amazon ECR                                            │
│  • SSH into EC2 + write .env + pull image                        │
│  • docker compose up -d + health check + auto-rollback           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Amazon ECR (Registry)                       │
│                    Docker Image Storage                          │
│                    idop-app:latest + idop-app:<sha>              │
└────────────────────────────┬────────────────────────────────────┘
                             │ docker compose pull
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AWS EC2 (Ubuntu 22.04 LTS)                     │
│  • Docker Compose orchestration                                  │
│  • idop-app container (FastAPI on port 8000)                     │
│  • checkpoint-db container (PostgreSQL 15)                       │
│  • Nginx reverse proxy (port 80/443 → 8000)                     │
│  • Let's Encrypt SSL (auto-renew)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
         ┌──────────────────┐  ┌────────────────┐
         │   AWS S3 Bucket  │  │ External APIs  │
         │ Document Cache   │  │ • OpenAI       │
         │                  │  │ • Voyage AI    │
         └──────────────────┘  │ • Tavily       │
                               │ • Qdrant Cloud │
                               │ • Upstash Redis│
                               │ • Supabase     │
                               └────────────────┘
```

---

## 2. Prerequisites & Requirements

### 2.1 Prerequisites Checklist

Before starting, ensure you have:

**Local Environment:**
- [ ] AWS CLI v2.x installed and configured
- [ ] Git installed
- [ ] GitHub account with repository access
- [ ] Docker Desktop installed (optional, for local testing)

**AWS Account:**
- [ ] Active AWS account with billing enabled
- [ ] Admin access or permissions for ECR, EC2, IAM, S3, CloudWatch
- [ ] An EC2 key pair `.pem` file for SSH access

**External Service Accounts (obtain before starting):**
- [ ] **OpenAI API Key** — https://platform.openai.com/api-keys
- [ ] **Qdrant Cloud Account + API Key** — https://cloud.qdrant.io/
- [ ] **Voyage AI API Key** — https://dash.voyageai.com/
- [ ] **Tavily Search API Key** — https://tavily.com/
- [ ] **Supabase Account + PostgreSQL Database** — https://supabase.com/
- [ ] (Optional) **Upstash Redis** for query caching — https://console.upstash.com/
- [ ] (Optional) **AWS S3 Bucket** for document caching

**Estimated Time:** 60-90 minutes for complete first-time setup

---

## 3. AWS IAM & EC2 Setup

### 3.1 Create IAM User for Deployment

1. Log into AWS Console: https://console.aws.amazon.com/
2. Navigate to **IAM** → **Users** → **Create user**
3. User name: `idop-deploy-user`
4. Select **Programmatic access**
5. Attach `AdministratorAccess` policy (or create a scoped policy — see Section 3.2)
6. Click **Create user**

**Generate Access Keys:**
1. Click on the created user → **Security credentials** tab
2. Click **Create access key** → Select "Command Line Interface (CLI)"
3. **Download the CSV** or copy:
   - **Access Key ID** (starts with `AKIA...`)
   - **Secret Access Key** (long random string)

⚠️ **CRITICAL:** Save these credentials securely. You won't be able to see the Secret Access Key again!

### 3.2 Minimum IAM Permissions (Recommended)

Instead of `AdministratorAccess`, you can create a scoped deployment policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRPermissions",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:DescribeImages"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3Permissions",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::idop-cache-*",
        "arn:aws:s3:::idop-cache-*/*"
      ]
    }
  ]
}
```

### 3.3 Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name idop-app \
  --region us-east-1 \
  --image-scanning-configuration scanOnPush=false \
  --encryption-configuration encryptionType=AES256
```

**Save the `repositoryUri`:**
```bash
export ECR_URI=$(aws ecr describe-repositories \
  --repository-names idop-app \
  --region us-east-1 \
  --query 'repositories[0].repositoryUri' \
  --output text)

echo "ECR Repository URI: $ECR_URI"
# Example: 221691784496.dkr.ecr.us-east-1.amazonaws.com/idop-app
```

### 3.4 Create S3 Bucket for Document Cache (Optional)

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET_NAME="idop-cache-${AWS_ACCOUNT_ID}"

aws s3 mb s3://${S3_BUCKET_NAME} --region us-east-1

echo "S3 Bucket: $S3_BUCKET_NAME"
```

---

## 4. External Services Configuration

### 4.1 OpenAI API Key

1. Go to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Name it: `IDOP Production`
4. Copy the key (starts with `sk-proj-...`)

**Expected Monthly Cost:** ~$5-50/month depending on usage

### 4.2 Qdrant Cloud Setup

1. Go to https://cloud.qdrant.io/
2. Create a new **Free Tier Cluster**
   - **Name:** `idop-production`
   - **Region:** AWS `us-east-1` or closest to your EC2
   - **Configuration:** 1GB RAM (free tier)
3. After cluster creation, get:
   - **Cluster URL:** `https://xxxxxxxx-xxxx.aws.cloud.qdrant.io`
   - **API Key:** Click "API Keys" → Generate new key

**⚠️ Create the collection manually (if not auto-created by the app):**
```bash
curl -X PUT "https://YOUR-CLUSTER.aws.cloud.qdrant.io/collections/idop_documents" \
  -H "api-key: YOUR_QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "dense": {
        "size": 1536,
        "distance": "Cosine"
      }
    },
    "sparse_vectors": {
      "sparse": {}
    }
  }'
```

**Expected Monthly Cost:** Free tier (1GB) or $25/month for production

### 4.3 Voyage AI API Key

1. Go to https://dash.voyageai.com/
2. Sign up / Log in
3. Navigate to **API Keys** → **Create Key**
4. Copy the key

**Expected Monthly Cost:** Free tier (50M tokens) or pay-per-use

### 4.4 Tavily Search API Key

1. Go to https://tavily.com/
2. Sign up / Log in
3. Navigate to **API Keys**
4. Copy the key (starts with `tvly-...`)

**Expected Monthly Cost:** Free tier (1000 searches/month) or $50/month

### 4.5 Supabase Database Setup

1. Go to https://supabase.com/ → Create new project
2. Configuration:
   - **Project Name:** `idop-production`
   - **Database Password:** (generate strong password and save it!)
   - **Region:** Closest to your EC2 instance
3. After creation, go to **Project Settings** → **Database**
4. Copy the **Session Pooler** connection string:
   ```
   postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
   ```

⚠️ **Important:** For the IDOP application, format the URL with the `psycopg` driver:
```
postgresql+psycopg://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:5432/postgres
```

**Expected Monthly Cost:** Free tier (500 MB) or $25/month Pro

### 4.6 Upstash Redis Setup (Optional)

1. Go to https://console.upstash.com/
2. Click **Create Database**
3. Configuration:
   - **Name:** `idop-query-cache`
   - **Type:** Regional
   - **Region:** US-East-1 (closest to EC2)
   - **TLS:** Enabled
4. After creation, copy:
   - **UPSTASH_REDIS_REST_URL** (e.g., `https://suited-mantis-12345.upstash.io`)
   - **UPSTASH_REDIS_REST_TOKEN**

**Expected Monthly Cost:** Free tier (10K requests/day) or $0.20 per 100K requests

---

## 5. GitHub Repository & Secrets Setup

### 5.1 Navigate to Repository Settings

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/IDOP`
2. Click **Settings** (top navigation bar)
3. In the left sidebar, click **Secrets and variables** → **Actions**

### 5.2 Add Required GitHub Secrets

Click **"New repository secret"** for **each** of the following secrets. The table below lists every secret the CI/CD pipeline uses, what value to put, where to get it, and which workflow file references it.

---

#### 🔐 Category 1: AWS / ECR Cloud Credentials

These secrets authenticate the GitHub Actions runner with your AWS account and ECR registry.

| Secret Name | Example Value | Where to Get It | Used By |
|:---|:---|:---|:---|
| `AWS_ACCESS_KEY_ID` | `AKIAIOSFODNN7EXAMPLE` | IAM → Users → Security credentials → Access Keys (Section 3.1) | `cd.yml` |
| `AWS_SECRET_ACCESS_KEY` | `wJalrXUtnFEMI/K7MDENG/bPxRfi...` | IAM → Users → Security credentials → Access Keys (Section 3.1) | `cd.yml` |
| `ECR_REGISTRY` | `221691784496.dkr.ecr.us-east-1.amazonaws.com` | ECR → Repositories → Copy the **registry root URL** (without the repo name, without trailing `/`) | `cd.yml` |

**How to find your ECR Registry URL:**
```bash
# Run this command to get your registry URL:
aws ecr describe-repositories \
  --repository-names idop-app \
  --region us-east-1 \
  --query 'repositories[0].repositoryUri' \
  --output text

# Output: 221691784496.dkr.ecr.us-east-1.amazonaws.com/idop-app
# ⚠️ REMOVE the "/idop-app" part — only use the registry root:
#     221691784496.dkr.ecr.us-east-1.amazonaws.com
```

---

#### 🔐 Category 2: EC2 Server Connection

These secrets allow the CD pipeline to SSH into your EC2 instance and deploy containers.

| Secret Name | Example Value | Where to Get It | Used By |
|:---|:---|:---|:---|
| `EC2_HOST` | `13.233.14.23` | EC2 Console → Instances → Copy **Public IPv4 address** | `cd.yml` |
| `EC2_SSH_KEY` | `-----BEGIN RSA PRIVATE KEY-----\nMIIEow...` | The **entire contents** of your `.pem` key file | `cd.yml` |

**How to get EC2_HOST:**
```bash
# From AWS CLI:
aws ec2 describe-instances \
  --region us-east-1 \
  --query 'Reservations[*].Instances[*].PublicIpAddress' \
  --output text
```

**How to add EC2_SSH_KEY:**
1. Open your `.pem` key file (e.g., `idop-key.pem`) in a text editor
2. Copy the **entire contents** including:
   ```
   -----BEGIN RSA PRIVATE KEY-----
   MIIEowIBAAKCAQEA...
   ...many lines of base64...
   -----END RSA PRIVATE KEY-----
   ```
3. Paste the full content as the secret value in GitHub

⚠️ **CRITICAL:** Do NOT add any extra spaces, newlines, or formatting. Paste the raw PEM content exactly as-is.

---

#### 🔐 Category 3: Internal Database Credentials (Hardcoded — No Secret Needed)

The internal PostgreSQL container (`checkpoint-db`) now uses a **hardcoded password** that is baked directly into the CD pipeline. This is safe because `checkpoint-db` has **no exposed ports** — it is only reachable from inside the Docker network by the `idop-app` container.

| Property | Value |
|:---|:---|
| **Password** | `idop_checkpoint_2026` |
| **Source** | Hardcoded in `cd.yml` Step 11d (`.env` heredoc) and Step 11e (`docker-compose.yml` heredoc) |
| **Secret Needed?** | ❌ No — do NOT set `POSTGRES_PASSWORD` as a GitHub secret |

**How this flows into the application (CD pipeline does this automatically):**
```
.env file on EC2:
  POSTGRES_PASSWORD=idop_checkpoint_2026
  DATABASE_URL=postgresql://postgres:idop_checkpoint_2026@checkpoint-db:5432/idop_memories

docker-compose.yml:
  POSTGRES_PASSWORD: idop_checkpoint_2026
```

⚠️ If you ever need to change this password, update it in **both** locations inside `cd.yml`:
1. `.env` heredoc (Step 11d) — `POSTGRES_PASSWORD=` and `DATABASE_URL=`
2. `docker-compose.yml` heredoc (Step 11e) — `POSTGRES_PASSWORD:`

---

#### 🔐 Category 4: Application API Keys & Integration Secrets

These secrets are injected as environment variables into the production `.env` file on your EC2 instance during every deployment.

| Secret Name | Example Value | Where to Get It | Used By |
|:---|:---|:---|:---|
| `OPENAI_API_KEY` | `sk-proj-abc123...` | https://platform.openai.com/api-keys (Section 4.1) | `cd.yml` |
| `VOYAGE_API_KEY` | `pa-abc123...` | https://dash.voyageai.com/ → API Keys (Section 4.3) | `cd.yml` |
| `TAVILY_API_KEY` | `tvly-abc123...` | https://tavily.com/ → API Keys (Section 4.4) | `cd.yml` |
| `QDRANT_URL` | `https://xyz-abc.aws.cloud.qdrant.io` | Qdrant Cloud Dashboard → Cluster URL (Section 4.2) | `cd.yml` |
| `QDRANT_API_KEY` | `eyJhbGci...` | Qdrant Cloud Dashboard → API Keys (Section 4.2) | `cd.yml` |
| `SUPABASE_DB_URL` | `postgresql+psycopg://postgres.ref:pass@aws-0-us-east-1.pooler.supabase.com:5432/postgres` | Supabase → Project Settings → Database → Session Pooler (Section 4.5) | `cd.yml` |

---

#### 🔐 Category 5: Optional Caching Secrets

These are optional. If not set, the application will run without caching (slightly slower on repeated queries).

| Secret Name | Example Value | Where to Get It | Used By |
|:---|:---|:---|:---|
| `UPSTASH_REDIS_URL` | `https://suited-mantis-12345.upstash.io` | Upstash Console → Database → REST URL (Section 4.6) | `cd.yml` |
| `UPSTASH_REDIS_TOKEN` | `AXN0ASQgODJjYTk...` | Upstash Console → Database → REST Token (Section 4.6) | `cd.yml` |
| `S3_CACHE_BUCKET` | `idop-cache-221691784496` | Your S3 bucket name from Section 3.4 | `cd.yml` |

---

### 5.3 Complete Secrets Summary Checklist

After adding all secrets, your GitHub **Settings → Secrets → Actions** page should look like this:

```
✅ Required Secrets (11 total):
┌─────────────────────────────┬──────────────────────────────────────┐
│ Secret Name                 │ Status                               │
├─────────────────────────────┼──────────────────────────────────────┤
│ AWS_ACCESS_KEY_ID           │ ● Added                              │
│ AWS_SECRET_ACCESS_KEY       │ ● Added                              │
│ ECR_REGISTRY                │ ● Added                              │
│ EC2_HOST                    │ ● Added                              │
│ EC2_SSH_KEY                 │ ● Added                              │
│ OPENAI_API_KEY              │ ● Added                              │
│ VOYAGE_API_KEY              │ ● Added                              │
│ TAVILY_API_KEY              │ ● Added                              │
│ QDRANT_URL                  │ ● Added                              │
│ QDRANT_API_KEY              │ ● Added                              │
│ SUPABASE_DB_URL             │ ● Added                              │
└─────────────────────────────┴──────────────────────────────────────┘

⬜ Optional Secrets (3 total):
┌─────────────────────────────┬──────────────────────────────────────┐
│ UPSTASH_REDIS_URL           │ ○ Optional (query caching)           │
│ UPSTASH_REDIS_TOKEN         │ ○ Optional (query caching)           │
│ S3_CACHE_BUCKET             │ ○ Optional (document caching)        │
└─────────────────────────────┴──────────────────────────────────────┘

ℹ️ POSTGRES_PASSWORD is no longer a required secret — it is hardcoded
   to idop_checkpoint_2026 in cd.yml (see Category 3 above).
```

### 5.4 How Secrets Flow into the Application

Understanding the full lifecycle of how secrets move from GitHub → EC2 → Container:

```
GitHub Repository Secrets
         │
         ▼ (referenced in cd.yml as ${{ secrets.XXX }})
┌──────────────────────────────────┐
│   CD Pipeline (cd.yml)           │
│                                  │
│   1. AWS_ACCESS_KEY_ID +         │
│      AWS_SECRET_ACCESS_KEY       │
│      → Configure AWS CLI         │
│      → Login to ECR              │
│                                  │
│   2. ECR_REGISTRY                │
│      → Tag & push Docker image   │
│                                  │
│   3. EC2_HOST + EC2_SSH_KEY      │
│      → SSH into EC2 instance     │
│                                  │
│   4. All API keys (except        │
│      POSTGRES_PASSWORD)          │
│      → Written to /home/ubuntu/  │
│        IDOP/.env on EC2 host     │
│                                  │
│     POSTGRES_PASSWORD is         │
│     HARDCODED directly in the    │
│     .env heredoc inside cd.yml   │
│     (not from GitHub Secrets)    │
└──────────────┬───────────────────┘
               │
               ▼ (SSH connection)
┌──────────────────────────────────┐
│   EC2 Instance                   │
│                                  │
│   /home/ubuntu/IDOP/.env         │
│   (chmod 600 — restricted)       │
│                                  │
│   Contains:                      │
│   ├── ENVIRONMENT=production     │
│   ├── ECR_REGISTRY=...          │
│   ├── POSTGRES_PASSWORD=       │
│   │   idop_checkpoint_2026       │
│   ├── DATABASE_URL=            │
│   │   postgresql://postgres:    │
│   │   idop_checkpoint_2026@...  │
│   ├── OPENAI_API_KEY=...        │
│   ├── VOYAGE_API_KEY=...        │
│   ├── TAVILY_API_KEY=...        │
│   ├── QDRANT_URL=...            │
│   ├── QDRANT_API_KEY=...        │
│   ├── UPSTASH_REDIS_URL=...     │
│   ├── UPSTASH_REDIS_TOKEN=...   │
│   ├── SUPABASE_DB_URL=...       │
│   └── S3_CACHE_BUCKET=...       │
└──────────────┬───────────────────┘
               │
               ▼ (docker compose up -d)
┌──────────────────────────────────┐
│   Docker Compose                 │
│                                  │
│   docker-compose.yml reads .env  │
│   and injects variables into:    │
│                                  │
│   ┌──────────────────────────┐   │
│   │ idop-app container       │   │
│   │ (FastAPI on port 8000)   │   │
│   │ Receives all API keys    │   │
│   │ from .env                │   │
│   └──────────────────────────┘   │
│                                  │
│   ┌──────────────────────────┐   │
│   │ checkpoint-db container  │   │
│   │ (PostgreSQL 15)          │   │
│   │ POSTGRES_PASSWORD is     │   │
│   │ hardcoded in compose     │   │
│   │ heredoc in cd.yml        │   │
│   └──────────────────────────┘   │
└──────────────────────────────────┘
```

### 5.5 Verify Workflow Configuration

Check that the workflows are correctly configured in your repository:

```bash
# Verify CI workflow exists and is correct
cat .github/workflows/ci.yml | head -10

# Expected output:
# name: IDOP Continuous Integration (CI)
# on:
#   push:
#     branches: [ main ]
#   pull_request:
#     branches: [ main ]
```

```bash
# Verify CD workflow exists and is correct
cat .github/workflows/cd.yml | head -15

# Expected output:
# name: IDOP Continuous Deployment (CD)
# on:
#   push:
#     branches: [ main ]
# env:
#   AWS_REGION: us-east-1
#   ECR_REPOSITORY: idop-app
#   EC2_PROJECT_PATH: /home/ubuntu/IDOP
```

---

## 6. EC2 Server Provisioning

### 6.1 Choose the Right Instance Specifications

| Setting | Value | Rationale |
|:---|:---|:---|
| **AMI** | Ubuntu Server 22.04 LTS (HVM), SSD, 64-bit x86 | Long-term support, Docker-compatible |
| **Instance Type** | `t3.medium` (2 vCPU, 4 GB RAM) minimum | Supports LangGraph + Docker Compose workloads |
| **Storage** | 30 GB gp3 SSD minimum | Docker images, Postgres WAL logs, chunk caches |
| **Key Pair** | Select or create a `.pem` key pair | Required for SSH access |
| **Region** | `us-east-1` (N. Virginia) | Must match your ECR region |

### 6.2 Configure Security Group Firewalls

Create a security group with these inbound rules:

| Protocol | Port Range | Source | Rationale |
|:---|:---|:---|:---|
| **TCP** | `22` | `My IP` / Bastion | Secure SSH terminal access |
| **TCP** | `80` | `0.0.0.0/0` | Let's Encrypt ACME verification & HTTP redirect |
| **TCP** | `443` | `0.0.0.0/0` | Secure HTTPS REST API calls from clients |
| **TCP** | `8000` | `EC2 Private IP` only | Internal FastAPI binding (blocked from public web) |

### 6.3 Allocate Elastic IP (Recommended)

To prevent your EC2_HOST from changing when the instance restarts:

```bash
# Allocate a static Elastic IP
aws ec2 allocate-address --region us-east-1

# Associate it with your instance
aws ec2 associate-address \
  --instance-id i-0abcdef1234567890 \
  --allocation-id eipalloc-0123456789abcdef0 \
  --region us-east-1
```

⚠️ If you use an Elastic IP, update the `EC2_HOST` GitHub secret with this static IP.

---

## 7. Docker & Environment Setup on EC2

> [!NOTE]
> **🚀 100% AUTOMATED STEP**
> You do **NOT** need to perform the SSH connection, Docker installation, Compose installation, AWS CLI setup, or directory creations manually anymore!
> The IDOP Continuous Deployment pipeline (`cd.yml`) automatically connects to your EC2 instance over SSH, checks for these dependencies, installs them if missing, configures your AWS ECR credentials dynamically, and writes your project directories/compose stacks on every deployment.
> 
> These manual instructions are provided below strictly for administrative reference, local maintenance, and custom server troubleshooting.

### 7.1 Connect to EC2 via SSH (Administrative Reference)

```bash
ssh -i "idop-key.pem" ubuntu@<EC2_HOST_IP>
```

### 7.2 Install Docker Engine & Compose

```bash
# Update Ubuntu package registers
sudo apt-get update -y
sudo apt-get upgrade -y

# Install prerequisite utilities
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Register Docker APT repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Compose
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add ubuntu user to docker group (run without sudo)
sudo usermod -aG docker $USER
```

**Exit the terminal and reconnect via SSH to refresh user group credentials.**

### 7.3 Install AWS CLI on EC2

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Configure AWS CLI with the same IAM credentials
aws configure
# AWS Access Key ID: AKIA...
# AWS Secret Access Key: wJalr...
# Default region: us-east-1
# Default output format: json

# Verify
aws sts get-caller-identity
```

### 7.4 Create Project Directory & Docker Compose File

```bash
mkdir -p /home/ubuntu/IDOP
cd /home/ubuntu/IDOP
```

Create the Docker Compose configuration:

```bash
cat << 'COMPOSEEOF' > docker-compose.yml
services:
  app:
    image: ${ECR_REGISTRY}/idop-app:latest
    container_name: idop-app
    restart: always
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      - ENV_STATE=production
      - DATABASE_URL=postgresql+psycopg://postgres:idop_checkpoint_2026@checkpoint-db:5432/idop_memories
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VOYAGE_API_KEY=${VOYAGE_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - UPSTASH_REDIS_URL=${UPSTASH_REDIS_URL}
      - UPSTASH_REDIS_TOKEN=${UPSTASH_REDIS_TOKEN}
      - SUPABASE_DB_URL=${SUPABASE_DB_URL}
      - S3_CACHE_BUCKET=${S3_CACHE_BUCKET}
    depends_on:
      checkpoint-db:
        condition: service_healthy
    networks:
      - idop-net

  checkpoint-db:
    image: postgres:16-alpine
    container_name: idop-checkpoint-db
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: idop_checkpoint_2026
      POSTGRES_DB: idop_memories
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres && psql -U postgres -d idop_memories -c 'SELECT 1' > /dev/null 2>&1"]
      interval: 5s
      timeout: 5s
      retries: 15
      start_period: 40s
    networks:
      - idop-net

volumes:
  pgdata:

networks:
  idop-net:
    name: idop-net
COMPOSEEOF
```

### 7.5 Create Initial .env File (One-Time Manual Setup)

For the first boot, create a temporary `.env` file manually. After the first GitHub Actions deployment, the CD pipeline will overwrite this automatically.

```bash
cat << 'EOF' > .env
# ─────────────────────────────────────────────────────────
# IDOP RUNTIME PRODUCTION SECRETS (MANUAL INITIAL SETUP)
# ─────────────────────────────────────────────────────────
# ⚠️ This file will be overwritten by CI/CD on every deployment.
# ⚠️ Only create this manually for the very first boot.
# ─────────────────────────────────────────────────────────

ENV_STATE=production
ECR_REGISTRY=221691784496.dkr.ecr.us-east-1.amazonaws.com
AWS_REGION=us-east-1

# POSTGRES_PASSWORD is hardcoded in cd.yml (idop_checkpoint_2026)
# Do NOT change below — the CD pipeline will overwrite this on deploy
POSTGRES_PASSWORD=idop_checkpoint_2026
OPENAI_API_KEY=sk-proj-your-key-here
VOYAGE_API_KEY=your-voyage-key-here
TAVILY_API_KEY=tvly-your-key-here
QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key-here
UPSTASH_REDIS_URL=https://your-redis.upstash.io
UPSTASH_REDIS_TOKEN=your-upstash-token-here
SUPABASE_DB_URL=postgresql+psycopg://postgres.ref:pass@host:5432/postgres
S3_CACHE_BUCKET=idop-cache-221691784496
EOF

# Secure the file
chmod 600 .env
```

---

## 8. Nginx Reverse Proxy & SSL

> [!NOTE]
> **🚀 100% AUTOMATED STEP**
> You do **NOT** need to manually install Nginx, write reverse proxy configuration blocks, or set up Certbot!
> The IDOP Continuous Deployment pipeline (`cd.yml`) automatically installs Nginx, configures the reverse proxy pointing to port 8000 with full SSE streaming support, binds to the EC2 Public IP address by default, and—if a `DOMAIN_NAME` repository secret is provided—installs Certbot and automatically acquires Let's Encrypt SSL certificates dynamically!
> 
> These manual instructions are provided below strictly for administrative reference, manual SSL adjustments, or DNS configuration troubleshooting.

### 8.1 Install Nginx & Certbot (Administrative Reference)

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
```

### 8.2 Configure Nginx Server Block

```bash
sudo rm /etc/nginx/sites-enabled/default

sudo cat << 'EOF' > /etc/nginx/sites-available/idop.conf
server {
    listen 80;
    server_name api.yourdomain.com;  # Replace with your actual domain

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Enable SSE Streaming support
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding on;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 24h;
    }
}
EOF

# Activate configuration
sudo ln -s /etc/nginx/sites-available/idop.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

### 8.3 Secure Domain with Let's Encrypt SSL

```bash
sudo certbot --nginx -d api.yourdomain.com \
  --non-interactive --agree-tos --email webmaster@yourdomain.com
```

Certbot will automatically configure SSL and redirect HTTP → HTTPS.

---

## 9. Triggering Deployment & Verification

### 9.1 Push to Main Branch

Once all GitHub Secrets are configured and the EC2 server is ready:

```bash
git add .
git commit -m "Deploy to production"
git push origin main
```

### 9.2 Monitor GitHub Actions

1. Go to your repository on GitHub
2. Click **Actions** tab
3. You should see two workflows running:
   - **IDOP Continuous Integration (CI)** — Lint, test, Docker build verification
   - **IDOP Continuous Deployment (CD)** — Build, push to ECR, deploy to EC2

### 9.3 What the CD Pipeline Does

The CD pipeline (`cd.yml`) executes these steps in order:

```
Step 1:  🧹 Free up disk space on GitHub runner
Step 2:  📥 Checkout code from repository
Step 3:  📋 Print deployment metadata (commit, actor, region)
Step 4:  🔐 Configure AWS credentials (AWS_ACCESS_KEY_ID + SECRET)
Step 5:  🔍 Verify AWS connection via STS identity check
Step 6:  🔑 Login to Amazon ECR
Step 7:  ⚙️ Set up QEMU (multi-platform support)
Step 8:  ⚙️ Set up Docker Buildx
Step 9:  🔨 Build Docker image (linux/amd64) and push to ECR
         Tags: ECR_REGISTRY/idop-app:latest + ECR_REGISTRY/idop-app:<commit-sha>
Step 10: 📋 Print build summary
Step 11: 🔐 SSH into EC2 and execute deployment:
         a. Write all secrets into /home/ubuntu/IDOP/.env (chmod 600)
         b. Login Docker to ECR on the EC2 host
         c. Tag current running image as "stable" (rollback backup)
         d. Pull new image from ECR
         e. docker compose down → docker compose up -d
         f. Health check with 5 retries (5 sec intervals)
         g. If health check fails → automatic rollback to stable tag
```

### 9.4 Post-Deployment Verification

**On your local machine:**
```bash
curl -f https://api.yourdomain.com/health
```

Expected response:
```json
{"status": "healthy", "timestamp": "2026-05-25T22:00:00Z"}
```

**On the EC2 server:**
```bash
# Check running containers
docker ps

# View application logs
docker logs -f idop-app

# View database logs
docker logs -f idop-checkpoint-db
```

### 9.5 Production Log Auditing

```bash
# Follow live application logs
docker logs -f idop-app

# View last 100 lines
docker logs --tail 100 idop-app

# Check database connection status
docker logs -f idop-checkpoint-db

# Clear cached states
curl -X POST "https://api.yourdomain.com/cache/clear" \
  -H "Authorization: Bearer <your-admin-token>"
```

---

## 10. Troubleshooting

### 10.1 GitHub Actions Deployment Fails

**Problem:** CD pipeline fails at "Configure AWS credentials"
**Solution:**
1. Verify `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are correct in GitHub Secrets
2. Check that the IAM user has ECR permissions
3. Ensure the region in `cd.yml` matches your ECR repository region

### 10.2 ECR Login Fails

**Problem:** "denied: Your authorization token has expired"
**Solution:** Re-run the failed GitHub Actions workflow. ECR tokens are temporary (12 hours).

### 10.3 SSH Connection Fails

**Problem:** CD pipeline fails at "Remote Deploy via SSH"
**Solution:**
1. Verify `EC2_HOST` IP is correct (check if it changed after instance reboot)
2. Verify `EC2_SSH_KEY` contains the full PEM file including headers
3. Check EC2 security group allows inbound SSH (port 22) from GitHub Actions IPs
4. Verify the EC2 instance is running: `aws ec2 describe-instance-status --instance-id i-xxx`

### 10.4 Health Check Fails After Deploy

**Problem:** Health check retries exhausted, automatic rollback triggered
**Solution:**
1. SSH into EC2: `ssh -i "key.pem" ubuntu@<EC2_HOST>`
2. Check container logs: `docker logs idop-app`
3. Common causes:
   - Missing or incorrect API keys in `.env`
   - Qdrant cluster not reachable (check URL)
   - Supabase database connection timeout
   - Port 8000 already in use

### 10.5 Docker Compose Pull Fails

**Problem:** "Error: image not found" when pulling from ECR
**Solution:**
1. Verify ECR login on EC2: `aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_REGISTRY>`
2. Verify the image exists: `aws ecr describe-images --repository-name idop-app --region us-east-1`

---

## Related Workflows

*   [13-service-initialization](./13-service-initialization.md) — Graph startup steps
*   [14-deployment](./14-deployment.md) — High-level multi-container orchestration
