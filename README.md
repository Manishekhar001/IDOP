# IDOP — E-commerce Data and Policy Assistant

[![CI](https://github.com/Manishekhar001/IDOP/actions/workflows/ci.yml/badge.svg)](https://github.com/Manishekhar001/IDOP/actions/workflows/ci.yml)

IDOP is a staff-facing assistant for exploring e-commerce records and finding answers in internal policy documents. Staff can ask questions in natural language, retrieve relevant policy guidance, and stage supported spreadsheet changes for review and approval.

It is a prototype intended for evaluation and controlled demos. It is not an autonomous order-management system and should not be connected to production data without a separate security, reliability, and operational review.

## What it demonstrates

| Capability | Example | Current boundary |
| --- | --- | --- |
| Business data questions | “How many customers are in each segment?” “Which products are low on stock?” | Read-only SQL over `customers`, `products`, and `orders` |
| Policy lookup | “What is the return window?” “How are delayed orders handled?” | Retrieves from the policy documents loaded into the configured Qdrant collection |
| Combined answers | “What is the return policy, and how many delivered orders are there?” | Can combine document retrieval and supported read-only data queries |
| Spreadsheet-assisted updates | Upload a supported change sheet, review its proposed changes, then approve | Only fields covered by `business_rules/rules.json`; approval flow is required |

The demo does not issue refunds, contact customers, modify payment records, or execute arbitrary natural-language writes.

## Synthetic demo data

The repository includes a small, fictional dataset for safe demonstrations:

- 8 customers, 10 products, and 20 orders in `demo_data/`.
- Three sample policy documents in `demo_data/policies/` covering returns, shipping, and inventory/pricing.
- All people, addresses, prices, orders, and policies are invented. The policy text is illustrative and is not legal or operational advice.

Orders in the current schema do not contain line items or delivery timestamps. This dataset can demonstrate basic customer, product, order-status, order-total, and policy questions; it cannot support product-level sales attribution or verify whether an individual order qualifies for a refund.

## Run the demo

### 1. Prepare dedicated demo services

You need Python 3.11+, PostgreSQL (a dedicated Supabase project or local PostgreSQL database), a dedicated Qdrant project for policy vectors, and credentials for the configured LLM and embedding providers.

Copy `.env.demo.example` to `.env`, then replace the placeholder API keys and endpoints. Set `DATABASE_URL`, `SUPABASE_DB_URL`, and `IDOP_DEMO_DATABASE_URL` to the same **dedicated demo database**. Do not point the seed command at a production or shared database. The sample local PostgreSQL URL works with the `postgres` service in `docker-compose.yml` after starting it with `docker compose up -d postgres`.

Set `QDRANT_URL` and `QDRANT_API_KEY` to a demo Qdrant project. The seed script uses the separate `idop_demo_documents` collection; configure `COLLECTION_NAME=idop_demo_documents` so the application reads those demo policy vectors.

### 2. Install and seed

```bash
python -m venv .venv
# Activate the virtual environment, then:
pip install -r requirements.txt
python scripts/seed_demo_data.py
```

The script requires the explicit `IDOP_DEMO_SEED_CONFIRM` value shown in `.env.demo.example`. It creates the supported business tables if needed, inserts the CSV data only when those tables are empty, and skips the SQL seed when this dataset has already been recorded. It does not drop tables or delete a Qdrant collection. `python scripts/seed_demo_data.py --database-only` skips policy indexing.

The script reads service credentials from `.env`, but this repository does not include credentials and does not seed any hosted service automatically.

### 3. Start the API

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API documentation. Register or sign in using the authentication routes before calling protected endpoints. For Docker, configure `.env` first and run `docker compose up --build`.

## Example demo questions

- “How many customers are in each segment?”
- “Show products with fewer than 20 units in stock.”
- “How many orders are delivered, pending, or cancelled?”
- “What does the sample policy say about return requests?”
- “Summarize low-stock products and the sample inventory guidance.”

Answers depend on the configured model, database contents, indexed documents, and the user's access. Review generated SQL and staged changes through the application's approval flow.

## How it works

1. The chat workflow classifies a question as a data query, document question, combined request, or general conversation.
2. Data questions use the relational schema and read-only SQL controls.
3. Document questions retrieve relevant passages from Qdrant and use them as context for an answer.
4. Supported spreadsheet changes are mapped, checked against business rules, and presented for approval before execution.

The API is built with FastAPI. PostgreSQL stores business data and application state; Qdrant stores document vectors. Provider and database settings are configured through environment variables in `.env`.

## Repository map

```text
app/                 API, authentication, chat workflows, SQL and document services
business_rules/      Rules for supported spreadsheet changes
config/               Application configuration and schema-related assets
demo_data/            Fictional CSV records and sample policy documents
scripts/              Database utilities and the opt-in demo seed command
tests/                Automated test suite
```

## Safety and demo data

- Seed only a dedicated, empty demo database. The seed script refuses to populate non-empty business tables unless its own dataset marker is present.
- Use dedicated demo credentials for the database, Qdrant, and model providers. Never commit `.env` or real customer data.
- Seeded customers and policies are fictional. Do not present sample policy text as an organization's actual policy.
- The application has authorization and approval flows, but that does not by itself establish production readiness. Review access controls, audit requirements, backups, monitoring, rate limits, and failure handling before any real deployment.

## Current limitations

- The included data model covers customers, products, and orders; it is not a complete commerce schema.
- Orders lack line items, delivery events, payment records, and return/refund state.
- The policy documents are examples. Retrieval quality depends on the documents and services configured by the operator.
- Spreadsheet changes are limited to the columns and validation rules implemented in `business_rules/rules.json`.
- This repository does not provide a hosted demo account or bundled credentials.

## Development

Install dependencies with `pip install -r requirements.txt`. Run the test suite with `pytest`. Keep secrets in local environment configuration and use synthetic data for development.

## License

See [LICENSE](LICENSE) for license terms.
