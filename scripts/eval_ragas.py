#!/usr/bin/env python3
"""
RAGAS Ablation Study — IDOP Pipeline Benchmarking Suite
========================================================

Runs a fixed 50-item benchmark dataset against 5 incremental pipeline
configurations, evaluates using the RAGAS framework (and an in-house LLM
evaluator as fallback), and produces a timestamped CSV/JSON report.

Pipeline configurations tested:

  1. Dense Only        — Dense vector search only (no hybrid, no HyDE, no reranking)
  2. Hybrid (RRF)      — Dense + sparse hybrid search with Reciprocal Rank Fusion
  3. Hybrid + HyDE     — Hybrid search + Hypothetical Document Embeddings query expansion
  4. Hybrid + Rerank   — Hybrid search + Voyage AI cross-encoder reranking
  5. Full CSRAG        — All features: HyDE + hybrid + reranking + CRAG + SRAG

Each configuration is a strict superset of the previous one (except Run 1→2
which switches search mode), isolating the marginal contribution of each
component.

Usage:
    # Default — run all 50 questions through all 5 configs
    python scripts/eval_ragas.py

    # Run only a subset (faster for iterative development)
    python scripts/eval_ragas.py --subset 10

    # Skip actual pipeline execution (use saved results from last run)
    python scripts/eval_ragas.py --reuse-last

    # Skip RAGAS dependency — use in-house LLM evaluator
    python scripts/eval_ragas.py --no-ragas

    # Custom output directory
    python scripts/eval_ragas.py --output-dir data/ablation_results

Output:
    data/ablation_results/
    ├── ablation_<timestamp>.json       # Full results (all configs × questions)
    ├── ablation_<timestamp>.csv        # Summary table per configuration
    ├── ablation_report_<timestamp>.txt  # Human-readable report
    └── latest                       → symlink to the most recent run directory
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Dataset — 50 questions across 5 categories
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK = [
    # ── Category 1: Refund & Customer Support Policies (10 questions) ──────
    {
        "question": "What is the standard refund window for retail products?",
        "ground_truth": "Retail products are eligible for a full refund within 30 days of purchase with receipt.",
        "category": "Refund & Support",
    },
    {
        "question": "How are international refund shipping costs handled?",
        "ground_truth": "For international orders, customers must cover return shipping costs unless the product arrived damaged.",
        "category": "Refund & Support",
    },
    {
        "question": "Are clearance items eligible for refunds?",
        "ground_truth": "All clearance items are marked as final sale and are strictly non-refundable.",
        "category": "Refund & Support",
    },
    {
        "question": "What is the processing time for approved digital credits?",
        "ground_truth": "Approved digital credits are processed within 24 to 48 business hours of verification.",
        "category": "Refund & Support",
    },
    {
        "question": "How long does a bank transfer refund take to reflect?",
        "ground_truth": "Direct bank transfer refunds typically take 5 to 7 business days to post to the account.",
        "category": "Refund & Support",
    },
    {
        "question": "Can gift cards be returned for cash refunds?",
        "ground_truth": "Gift cards cannot be returned, exchanged, or redeemed for cash refunds under any circumstances.",
        "category": "Refund & Support",
    },
    {
        "question": "What is the policy for restocking fees on opened electronics?",
        "ground_truth": "Opened electronics are subject to a 15% restocking fee if returned within the 30-day window.",
        "category": "Refund & Support",
    },
    {
        "question": "What happens if a package is lost in transit by the carrier?",
        "ground_truth": "Lost in transit claims must be filed within 14 days, and the company will issue a replacement or full refund.",
        "category": "Refund & Support",
    },
    {
        "question": "Are shipping fees refundable on voluntary returns?",
        "ground_truth": "Original shipping and handling fees are non-refundable for voluntary customer returns.",
        "category": "Refund & Support",
    },
    {
        "question": "What is the return window for promotional bundle items?",
        "ground_truth": "Promotional bundle items must be returned together to receive a full refund; partial returns are rejected.",
        "category": "Refund & Support",
    },
    # ── Category 2: Corporate & Administrative Policies (10 questions) ──────
    {
        "question": "What is the standard office core hours requirement?",
        "ground_truth": "Core collaboration hours are from 10:00 AM to 3:00 PM EST daily.",
        "category": "Corporate Policy",
    },
    {
        "question": "How is the annual health wellness stipend claimed?",
        "ground_truth": "Wellness stipends of up to $500 are claimed by submitting receipts through the HR Expense portal before December 1st.",
        "category": "Corporate Policy",
    },
    {
        "question": "What is the default employee referral bonus payout?",
        "ground_truth": "The standard referral bonus is $2,000, paid in two installments after the new hire completes 90 and 180 days.",
        "category": "Corporate Policy",
    },
    {
        "question": "What are the rules for travel flight class booking approvals?",
        "ground_truth": "All business flights under 6 hours must be booked in economy class. Business class requires VP approval.",
        "category": "Corporate Policy",
    },
    {
        "question": "How many consecutive days of paid sick leave require a doctor note?",
        "ground_truth": "Consecutive sick leaves of 3 or more days require a valid medical certificate submitted to HR.",
        "category": "Corporate Policy",
    },
    {
        "question": "What is the equipment return timeline upon employee departure?",
        "ground_truth": "Departing employees must return all company-owned hardware within 5 business days of their final day.",
        "category": "Corporate Policy",
    },
    {
        "question": "What is the maximum corporate gift value employees can accept?",
        "ground_truth": "Employees cannot accept corporate gifts exceeding a nominal value of $100 without compliance sign-off.",
        "category": "Corporate Policy",
    },
    {
        "question": "How is the tuition reimbursement program structured?",
        "ground_truth": "The company reimburses up to $5,250 annually for pre-approved, job-related graduate coursework with grade B or higher.",
        "category": "Corporate Policy",
    },
    {
        "question": "What is the standard parental leave benefit duration?",
        "ground_truth": "The platform provides 12 weeks of fully paid parental leave for primary and secondary caregivers after 1 year of service.",
        "category": "Corporate Policy",
    },
    {
        "question": "Are contract workers eligible for dental insurance plans?",
        "ground_truth": "Contract workers are generally ineligible for corporate group dental benefits unless explicitly detailed in their agreement.",
        "category": "Corporate Policy",
    },
    # ── Category 3: Database & SQL Inquiries (10 questions) ─────────────────
    {
        "question": "What is the primary table to retrieve active subscriber records?",
        "ground_truth": "Active subscriber records are stored inside the 'subscriptions' table where 'status' equals 'active'.",
        "category": "Database Schema",
    },
    {
        "question": "Which column holds employee salary details in the employees schema?",
        "ground_truth": "Salary details are stored in the 'salary' column of the 'employees' table as a numeric type.",
        "category": "Database Schema",
    },
    {
        "question": "How is a database query for top 5 products by order volume structured?",
        "ground_truth": "Query SELECT product_id, SUM(quantity) FROM order_items GROUP BY product_id ORDER BY SUM(quantity) DESC LIMIT 5;",
        "category": "Database Schema",
    },
    {
        "question": "What is the table representing database security audit logs?",
        "ground_truth": "Security audit logs are stored in the 'audit_logs' database table containing query and mutation signatures.",
        "category": "Database Schema",
    },
    {
        "question": "Which table links orders to customer profiles?",
        "ground_truth": "The 'orders' table contains a 'customer_id' foreign key linking directly to the 'customers' table.",
        "category": "Database Schema",
    },
    {
        "question": "How is employee department allocation mapped in the database?",
        "ground_truth": "Employee department mapping uses the 'employee_departments' junction table linking 'employee_id' and 'department_id'.",
        "category": "Database Schema",
    },
    {
        "question": "What is the column tracking cryptographically signed approval tokens?",
        "ground_truth": "Cryptographic tokens are mapped in the 'approval_token' string column inside the 'pending_transactions' register table.",
        "category": "Database Schema",
    },
    {
        "question": "What table maps product inventory count across warehouses?",
        "ground_truth": "Product stock levels are managed in the 'inventory' table keyed by 'product_id' and 'warehouse_id'.",
        "category": "Database Schema",
    },
    {
        "question": "Which field logs transaction status during mutation attempts?",
        "ground_truth": "The mutation transaction status is tracked in the 'status' enum column mapping to 'pending', 'executed', or 'failed'.",
        "category": "Database Schema",
    },
    {
        "question": "What represents the main catalog table for store listings?",
        "ground_truth": "Listings are stored inside the 'catalog_items' table containing pricing, descriptors, and stock identifiers.",
        "category": "Database Schema",
    },
    # ── Category 4: HR, Roles & Workspace Operations (10 questions) ─────────
    {
        "question": "What is the threshold limit for bulk rows uploads in mutations?",
        "ground_truth": "The platform enforces a strict threshold limit of 1000 rows per spreadsheet mutation upload to protect database memory.",
        "category": "Platform Operations",
    },
    {
        "question": "What is the default department enum values allowed?",
        "ground_truth": "Allowed corporate department enums are 'HR', 'Engineering', 'Sales', and 'Finance'.",
        "category": "Platform Operations",
    },
    {
        "question": "Where are business rule configurations declared?",
        "ground_truth": "Validation constraints are declared in the local 'business_rules/rules.json' configuration file.",
        "category": "Platform Operations",
    },
    {
        "question": "What is the maximum allowed salary for a junior analyst role?",
        "ground_truth": "According to rules.json, the maximum permitted salary for junior tiers is capped at $120,000.",
        "category": "Platform Operations",
    },
    {
        "question": "Who must approve a transaction mutation after rule validation?",
        "ground_truth": "All validated mutations require human-in-the-loop validation using a secure token at the '/mutation/approve' route.",
        "category": "Platform Operations",
    },
    {
        "question": "How are empty spreadsheet cells treated in rule checking?",
        "ground_truth": "Empty or null cells are rejected with validation errors if the rule declares the column as required/non-null.",
        "category": "Platform Operations",
    },
    {
        "question": "What is the default thread checkpoint save interval?",
        "ground_truth": "The LangGraph state checkpointer commits checkpoints instantly upon transition of every active processing node.",
        "category": "Platform Operations",
    },
    {
        "question": "What database container engine is active for checkpointer data?",
        "ground_truth": "IDOP uses an internal PostgreSQL Docker container for checkpoint states and user facts stores.",
        "category": "Platform Operations",
    },
    {
        "question": "Which external API handles semantic query reranking?",
        "ground_truth": "Semantic query reranking is offloaded to the Voyage AI Rerank-2.5 Cross-Encoder API.",
        "category": "Platform Operations",
    },
    {
        "question": "What dense vector dimension size is active in Qdrant collections?",
        "ground_truth": "Qdrant utilizes 1536-dimensional dense vectors generated by OpenAI text-embedding-3-small.",
        "category": "Platform Operations",
    },
    # ── Category 5: Complex Hybrid Scenarios (10 questions) ─────────────────
    {
        "question": "If an international user returned opened electronics, what is the fee policy?",
        "ground_truth": "International returns require the customer to cover shipping, and opened electronics are subject to a 15% restocking fee.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "What happens if a VP wants to book a flight over 6 hours?",
        "ground_truth": "Flights over 6 hours do not have class restrictions, but standard flights under 6 hours require economy class unless approved by VP.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "How are customer support refunds on final clearance sales items evaluated?",
        "ground_truth": "Customer support agents must reject refunds on final sale clearance items because they are strictly non-refundable.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "Does a contractor referral bonus follow the same payout timeframe?",
        "ground_truth": "Referral bonuses apply only to full-time hires; contract worker referrals are ineligible for standard HR referral payouts.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "What is the database column to check if an employee has completed equipment return?",
        "ground_truth": "Check 'returned_at' timestamp inside 'assets_log' where employee_id matches and state equals 'returned'.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "Does a bank transfer refund on opened electronics include original shipping?",
        "ground_truth": "Original shipping is non-refundable on voluntary returns, and opened electronics are docked 15% restocking fee.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "If an active subscriber claims lost package, does the platform auto-replace?",
        "ground_truth": "The platform issues replacement or refund after loss verification, provided the claim is filed within 14 days of shipment.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "Can wellness stipend submit flight receipts for reimbursement?",
        "ground_truth": "No, wellness stipends of up to $500 are reserved for health/wellness expenses. Travel flight expenses follow corporate travel claims.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "If a referral resigns after 100 days, is the entire bonus paid?",
        "ground_truth": "No, only the first referral installment is paid at 90 days. The second installment at 180 days is voided if they resign.",
        "category": "Multi-hop Reasoning",
    },
    {
        "question": "What is the validation action if upload contains an unmapped column?",
        "ground_truth": "The ColumnMapper matches headings using GPT-4o-mini; any entirely unmapped or irrelevant columns are ignored or raise a schema error.",
        "category": "Multi-hop Reasoning",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configurations — Incremental ablation design
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Dense Only",
        "description": "Baseline — single dense vector similarity search (OpenAI text-embedding-3-small)",
        "search_mode": "dense",
        "enable_hyde": False,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 2,
        "name": "Hybrid (RRF)",
        "description": "Dense + sparse hybrid search with Qdrant Reciprocal Rank Fusion (k=60)",
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 3,
        "name": "Hybrid + HyDE",
        "description": "Hybrid search + Hypothetical Document Embeddings query expansion (GPT-4o-mini, 3 hypotheses)",
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 4,
        "name": "Hybrid + Reranking",
        "description": "Hybrid search + Voyage AI Rerank-2.5 cross-encoder reranking",
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_reranking": True,
        "top_k": 5,
    },
    {
        "id": 5,
        "name": "Full CSRAG",
        "description": "All features: HyDE + hybrid search + reranking + CRAG evaluation + SRAG verification + context enrichment",
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_reranking": True,
        "top_k": 5,
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Simple ASCII progress bar."""
    filled = int(width * current / total)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {current}/{total}"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════════

_LAST_RESULTS_PATH: Optional[Path] = None  # set after a successful run


async def _pg_connect_and_setup(
    cls, database_url: str, name: str, max_retries: int = 3, delay: float = 1.0
):
    """
    Create a Postgres async pool, enter context, run migrations, with retry.

    Freshly-restarted Postgres containers sometimes close the first few
    connections during background recovery. Retrying resolves this.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        resource = None
        try:
            resource = await cls.from_conn_string(database_url).__aenter__()
            await resource.setup()
            if attempt > 1:
                print(f"  [OK] {name} connected on retry attempt {attempt}")
            return resource
        except Exception as e:
            last_exc = e
            print(
                f"  [WARN] {name} attempt {attempt}/{max_retries} failed: "
                f"{type(e).__name__}: {e}"
            )
            if resource is not None:
                try:
                    await resource.__aexit__(None, None, None)
                except Exception:
                    pass
            if attempt < max_retries:
                delay_secs = delay * (2 ** (attempt - 1))
                print(f"  [INFO] Retrying {name} in {delay_secs}s...")
                await asyncio.sleep(delay_secs)
    raise last_exc  # type: ignore[misc]


def _collect_contexts(state: dict) -> List[str]:
    """Extract context texts from a pipeline result state dict."""
    from langchain_core.documents import Document

    contexts: List[str] = []
    for key in ("good_docs", "docs", "web_docs"):
        docs = state.get(key, []) or []
        for d in docs:
            if isinstance(d, Document):
                contexts.append(d.page_content)
            elif isinstance(d, dict):
                contexts.append(d.get("content", d.get("page_content", "")))
    return contexts


async def run_single_query(
    question: str,
    config: Dict[str, Any],
    config_dir: Path,
) -> Dict[str, Any]:
    """
    Run one question through one pipeline configuration.

    Returns a dict with question, answer, contexts, ground_truth, and timing.
    """
    from app.core.csrag_engine import CSRAGEngine
    from app.core.vector_store import VectorStoreService
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from app.config import get_settings

    settings = get_settings()

    # Initialize services (with retry for Postgres startup race)
    vector_store = VectorStoreService()
    store = await _pg_connect_and_setup(
        AsyncPostgresStore,
        settings.database_url,
        "AsyncPostgresStore",
        max_retries=5,
        delay=1.0,
    )
    checkpointer = await _pg_connect_and_setup(
        AsyncPostgresSaver,
        settings.database_url,
        "AsyncPostgresSaver",
        max_retries=5,
        delay=1.0,
    )

    try:
        engine = CSRAGEngine(vector_store, store, checkpointer)
        thread_id = f"ablation-cfg{config['id']}-{config_dir.name}"

        t0 = time.monotonic()
        res = await engine.aquery(
            question=question,
            thread_id=thread_id,
            user_id="ablation-eval",
            search_mode=config["search_mode"],
            top_k=config["top_k"],
            enable_hyde=config["enable_hyde"],
            enable_reranking=config["enable_reranking"],
        )
        elapsed = time.monotonic() - t0

        answer = res.get("answer", "")
        contexts = _collect_contexts(res)

        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "pipeline_time_s": round(elapsed, 3),
            "crag_verdict": res.get("crag_verdict", ""),
            "issup": res.get("issup", ""),
            "isuse": res.get("isuse", ""),
        }

    finally:
        await store.__aexit__(None, None, None)
        await checkpointer.__aexit__(None, None, None)


async def run_configuration(
    config: Dict[str, Any],
    benchmark: List[Dict[str, Any]],
    config_dir: Path,
    output_dir: Path,
    use_ragas: bool,
) -> Dict[str, Any]:
    """
    Run one ablation configuration against all benchmark questions.

    Returns aggregated results dict:
        config_id, name, description, metrics {faithfulness, ...}, per-question list, timing
    """
    cfg_id = config["id"]
    cfg_name = config["name"]
    n_questions = len(benchmark)

    print(f"\n{'=' * 70}")
    print(f"  Run {cfg_id}/5: {cfg_name}")
    print(f"  {config['description']}")
    print(
        f"  search_mode={config['search_mode']}, hyde={config['enable_hyde']}, "
        f"rerank={config['enable_reranking']}, top_k={config['top_k']}"
    )
    print(f"{'=' * 70}")

    # ── 1. Execute all queries ──────────────────────────────────────────
    pipeline_outputs: List[Dict[str, Any]] = []
    start_time = time.monotonic()

    for q_idx, item in enumerate(benchmark, start=1):
        progress = _progress_bar(q_idx, n_questions)
        short_q = item["question"][:70]
        print(f"  {progress}  {short_q}...", end="", flush=True)

        try:
            out = await run_single_query(item["question"], config, config_dir)
            out["ground_truth"] = item["ground_truth"]
            out["category"] = item.get("category", "")
            out["config_id"] = cfg_id
            pipeline_outputs.append(out)
            print(f"  OK ({out['pipeline_time_s']:.1f}s)")
        except Exception as exc:
            print(f"  FAIL ERROR: {type(exc).__name__}: {exc}")
            pipeline_outputs.append(
                {
                    "question": item["question"],
                    "ground_truth": item["ground_truth"],
                    "category": item.get("category", ""),
                    "config_id": cfg_id,
                    "answer": f"[ERROR] {type(exc).__name__}: {exc}",
                    "contexts": [],
                    "pipeline_time_s": 0.0,
                    "crag_verdict": "",
                    "issup": "",
                    "isuse": "",
                }
            )

    total_time = time.monotonic() - start_time
    avg_time = round(total_time / n_questions, 3)

    # ── 2. Evaluate with RAGAS ──────────────────────────────────────────
    metrics: Dict[str, float] = {}
    ragas_used = False

    if use_ragas:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            data = {
                "question": [o["question"] for o in pipeline_outputs],
                "answer": [o["answer"] for o in pipeline_outputs],
                "contexts": [o["contexts"] for o in pipeline_outputs],
                "ground_truth": [o["ground_truth"] for o in pipeline_outputs],
            }
            dataset = Dataset.from_dict(data)

            score_res = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )
            # RAGAS 0.4.x EvaluationResult has .scores dict but not .get()
            scores_dict = score_res.scores
            metrics = {
                "faithfulness": round(float(scores_dict.get("faithfulness", 0.0)), 4),
                "answer_relevancy": round(
                    float(scores_dict.get("answer_relevancy", 0.0)), 4
                ),
                "context_precision": round(
                    float(scores_dict.get("context_precision", 0.0)), 4
                ),
                "context_recall": round(
                    float(scores_dict.get("context_recall", 0.0)), 4
                ),
            }
            ragas_used = True
            print(
                f"\n  [RAGAS] Evaluation complete — "
                f"faith={metrics['faithfulness']:.3f}, "
                f"relev={metrics['answer_relevancy']:.3f}, "
                f"prec={metrics['context_precision']:.3f}, "
                f"recall={metrics['context_recall']:.3f}"
            )

        except ImportError:
            print(
                "\n  [SKIP] RAGAS package not installed — will use in-house evaluator"
            )
        except Exception as exc:
            print(f"\n  [RAGAS ERROR] {type(exc).__name__}: {exc} — falling back")

    # ── 3. If RAGAS unavailable/failed, use in-house evaluator ───────────
    if not ragas_used:
        metrics = await _evaluate_in_house(pipeline_outputs)

    # Ensure all 4 metrics are present (in-house evaluator may not compute recall)
    metrics.setdefault("context_recall", 0.0)

    # ── 4. Compute per-category breakdown ───────────────────────────────
    cat_metrics = _compute_category_metrics(pipeline_outputs)

    result = {
        "config_id": cfg_id,
        "name": cfg_name,
        "description": config["description"],
        "parameters": {
            "search_mode": config["search_mode"],
            "enable_hyde": config["enable_hyde"],
            "enable_reranking": config["enable_reranking"],
            "top_k": config["top_k"],
        },
        "metrics": metrics,
        "category_metrics": cat_metrics,
        "ragas_used": ragas_used,
        "timing": {
            "total_seconds": round(total_time, 3),
            "avg_seconds_per_query": avg_time,
        },
        "questions": pipeline_outputs,
    }

    # ── 5. Save per-configuration results ───────────────────────────────
    cfg_file = (
        config_dir
        / f"config_{cfg_id:02d}_{config['name'].replace(' ', '_').lower()}.json"
    )
    with open(cfg_file, "w", encoding="utf-8") as f:
        # Strip bulky per-question data from the main file to keep it lightweight
        summary = {k: v for k, v in result.items() if k != "questions"}
        summary["question_count"] = len(pipeline_outputs)
        json.dump(summary, f, indent=2, default=str)
    print(f"  [SAVED] {cfg_file.name}")

    return result


async def _evaluate_in_house(
    pipeline_outputs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Use the in-house RagasEvaluator (LLM-based) as fallback."""
    print("  [EVAL] Running in-house RagasEvaluator on all responses...")
    from app.core.feature3_rag.ragas_evaluator import get_ragas_evaluator

    evaluator = get_ragas_evaluator()
    faith_scores: List[float] = []
    relev_scores: List[float] = []
    prec_scores: List[float] = []
    # The in-house RagasScores model doesn't include context_recall,
    # so compute it as a proxy: relevant_chunks / total_chunks averaged
    # across all evaluated questions.
    recall_numer: List[float] = []

    for out in pipeline_outputs:
        answer = out.get("answer", "")
        contexts = out.get("contexts", [])
        if not answer or answer.startswith("[ERROR]"):
            continue

        try:
            scores = await evaluator.evaluate(
                question=out["question"],
                answer=answer,
                contexts=contexts,
            )
            if scores:
                faith_scores.append(scores.faithfulness)
                relev_scores.append(scores.answer_relevancy)
                prec_scores.append(scores.context_precision)
                # Proxy recall: ratio of relevant chunks to total chunks
                total = max(scores.context_total_count, 1)
                recall_numer.append(scores.context_relevant_count / total)
        except Exception:
            pass

    n = len(faith_scores) or 1
    metrics = {
        "faithfulness": round(sum(faith_scores) / n, 4) if faith_scores else 0.0,
        "answer_relevancy": round(sum(relev_scores) / n, 4) if relev_scores else 0.0,
        "context_precision": round(sum(prec_scores) / n, 4) if prec_scores else 0.0,
        "context_recall": round(sum(recall_numer) / n, 4) if recall_numer else 0.0,
    }
    print(
        f"  [EVAL] In-house complete — "
        f"faith={metrics['faithfulness']:.3f}, "
        f"relev={metrics['answer_relevancy']:.3f}, "
        f"prec={metrics['context_precision']:.3f}, "
        f"recall={metrics['context_recall']:.3f}"
    )
    return metrics


def _compute_category_metrics(
    outputs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-category success rates based on answer quality heuristics."""
    categories = defaultdict(list)
    for o in outputs:
        cat = o.get("category", "Unknown")
        answer = o.get("answer", "")
        has_error = answer.startswith("[ERROR]")
        has_answer = (
            bool(answer)
            and not has_error
            and answer != "I don't have enough information."
        )
        categories[cat].append(has_answer)

    cat_metrics = {}
    for cat, results in categories.items():
        total = len(results)
        success = sum(results)
        cat_metrics[cat] = {
            "total": total,
            "successful": success,
            "success_rate": round(success / total, 4) if total > 0 else 0.0,
        }
    return cat_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def _build_report(
    all_results: List[Dict[str, Any]],
    benchmark_name: str,
    total_elapsed: float,
) -> str:
    """Build a human-readable ablation report string."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  IDOP RAGAS ABLATION STUDY REPORT")
    lines.append(f"  Benchmark: {benchmark_name} ({len(BENCHMARK)} questions)")
    lines.append(f"  Completed: {_timestamp()}")
    lines.append(f"  Total time: {total_elapsed:.1f}s")
    lines.append("=" * 72)
    lines.append("")

    # ── Summary Table ───────────────────────────────────────────────────
    header = (
        f"{'Run':<5} {'Configuration':<30} {'Faith':<9} {'Relev':<9} "
        f"{'Prec':<9} {'Recall':<9} {'Avg Q(s)':<9}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    base_metrics = None
    for r in all_results:
        m = r["metrics"]
        imp_faith = ""
        if base_metrics and base_metrics.get("faithfulness", 0) > 0:
            delta = m["faithfulness"] - base_metrics["faithfulness"]
            imp_faith = f" ({delta:+.1%})"
        lines.append(
            f"{r['config_id']:<5} {r['name']:<30} "
            f"{m['faithfulness']:<9.4f}{imp_faith:<9} "
            f"{m['answer_relevancy']:<9.4f} "
            f"{m['context_precision']:<9.4f} "
            f"{m['context_recall']:<9.4f} "
            f"{r['timing']['avg_seconds_per_query']:<9.3f}"
        )
        if base_metrics is None:
            base_metrics = m

    lines.append("-" * len(header))
    if len(all_results) >= 2:
        first = all_results[0]["metrics"]
        last = all_results[-1]["metrics"]
        lines.append("")
        lines.append("  Cumulative Improvements (Full CSRAG vs Dense Only):")
        for key in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ):
            delta = last[key] - first[key]
            pct = (delta / first[key] * 100) if first[key] > 0 else 0
            lines.append(
                f"    {key:<25} {first[key]:.4f} -> {last[key]:.4f}  ({pct:+.1f}%)"
            )
    lines.append("")

    # ── Per-Configuration Details ────────────────────────────────────────
    for r in all_results:
        lines.append(f"\n{'-' * 50}")
        lines.append(f"  Configuration {r['config_id']}: {r['name']}")
        lines.append(f"  {r['description']}")
        lines.append(f"  Parameters: {json.dumps(r['parameters'])}")
        lines.append(f"  RAGAS used: {r.get('ragas_used', False)}")
        lines.append(f"  Avg query time: {r['timing']['avg_seconds_per_query']:.3f}s")
        lines.append("")
        m = r["metrics"]
        lines.append(f"    faithfulness       = {m['faithfulness']:.4f}")
        lines.append(f"    answer_relevancy    = {m['answer_relevancy']:.4f}")
        lines.append(f"    context_precision   = {m['context_precision']:.4f}")
        lines.append(f"    context_recall      = {m['context_recall']:.4f}")
        if r.get("category_metrics"):
            lines.append("")
            lines.append("    Per-Category Success Rates:")
            for cat, cm in sorted(r["category_metrics"].items()):
                lines.append(
                    f"      {cat:<25} {cm['successful']:>3}/{cm['total']:<3} "
                    f"({cm['success_rate']:.1%})"
                )

    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)


def _save_csv(results_dir: Path, all_results: List[Dict[str, Any]]) -> Path:
    """Save a CSV summary table."""
    csv_path = results_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config_id",
                "name",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "avg_query_time_s",
                "total_time_s",
            ]
        )
        for r in all_results:
            m = r["metrics"]
            writer.writerow(
                [
                    r["config_id"],
                    r["name"],
                    m["faithfulness"],
                    m["answer_relevancy"],
                    m["context_precision"],
                    m["context_recall"],
                    r["timing"]["avg_seconds_per_query"],
                    r["timing"]["total_seconds"],
                ]
            )
    return csv_path


def _save_per_question_csv(
    results_dir: Path, all_results: List[Dict[str, Any]]
) -> Path:
    """Save a detailed per-question × per-configuration CSV."""
    csv_path = results_dir / "per_question_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "config_id",
                "config_name",
                "question",
                "category",
                "has_answer",
                "has_error",
                "pipeline_time_s",
                "crag_verdict",
                "issup",
                "isuse",
            ]
        )
        for r in all_results:
            for q in r.get("questions", []):
                answer = q.get("answer", "")
                writer.writerow(
                    [
                        r["config_id"],
                        r["name"],
                        q["question"],
                        q.get("category", ""),
                        1 if bool(answer) and not answer.startswith("[ERROR]") else 0,
                        1 if answer.startswith("[ERROR]") else 0,
                        q.get("pipeline_time_s", 0.0),
                        q.get("crag_verdict", ""),
                        q.get("issup", ""),
                        q.get("isuse", ""),
                    ]
                )
    return csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IDOP RAGAS Ablation Study — benchmark 5 pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset",
        "-n",
        type=int,
        default=None,
        help="Run on a random subset of N questions (default: all 50)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/ablation_results",
        help="Output directory for results (default: data/ablation_results)",
    )
    parser.add_argument(
        "--reuse-last",
        action="store_true",
        help="Skip pipeline execution and re-report from the most recent results",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS library evaluation; use in-house LLM evaluator only",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset selection (default: 42)",
    )
    return parser.parse_args(argv)


async def main_async() -> None:
    """Async entry point."""
    import asyncio

    # Windows compatibility: psycopg async requires SelectorEventLoop
    # rather than the default ProactorEventLoop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    args = parse_args()
    use_ragas = not args.no_ragas
    benchmark = list(BENCHMARK)

    # ── Subset logic ────────────────────────────────────────────────────
    if args.subset and 0 < args.subset < len(benchmark):
        import random

        rng = random.Random(args.seed)
        rng.shuffle(benchmark)
        benchmark = benchmark[: args.subset]
        print(
            f"  [SUBSET] Using {len(benchmark)} questions (--subset={args.subset}, seed={args.seed})"
        )
    else:
        print(f"  [FULL] Using all {len(benchmark)} benchmark questions")

    # ── Output directory ────────────────────────────────────────────────
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_ts = _timestamp()
    config_dir = base_dir / f"run_{run_ts}"
    config_dir.mkdir(parents=True, exist_ok=True)

    # ── Reuse / fresh execution ─────────────────────────────────────────
    if args.reuse_last:
        print("\n  [REUSE] Skipping pipeline execution. Looking for last results...")
        runs = sorted(base_dir.glob("run_*"), reverse=True)
        if len(runs) < 2:
            print("  [ERROR] No previous run found for --reuse-last")
            sys.exit(1)
        # The current empty config_dir is the latest — use the one before it
        prev = runs[1]
        print(f"  [REUSE] Loading from {prev.name}")
        all_results = []
        for cfg in ABLATION_CONFIGS:
            cfg_file = (
                prev
                / f"config_{cfg['id']:02d}_{cfg['name'].replace(' ', '_').lower()}.json"
            )
            if cfg_file.exists():
                with open(cfg_file) as f:
                    all_results.append(json.load(f))
                print(f"    Loaded {cfg_file.name}")
            else:
                print(f"    [WARN] {cfg_file.name} not found - skipping")
        total_elapsed = 0.0
    else:
        print(f"\n  {'=' * 50}")
        print(f"  Starting ablation study — {len(ABLATION_CONFIGS)} configurations")
        print(f"  Output: {config_dir}")
        print(f"  RAGAS: {'enabled' if use_ragas else 'disabled (in-house evaluator)'}")
        print(f"  {'=' * 50}")

        all_results = []
        t_start = time.monotonic()
        for cfg in ABLATION_CONFIGS:
            result = await run_configuration(
                cfg, benchmark, config_dir, base_dir, use_ragas
            )
            all_results.append(result)
        total_elapsed = time.monotonic() - t_start

    # ── Generate report ─────────────────────────────────────────────────
    report = _build_report(
        all_results, f"IDOP Benchmark ({len(benchmark)} questions)", total_elapsed
    )
    report_path = config_dir / "ablation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Save CSVs ────────────────────────────────────────────────────────
    _save_csv(config_dir, all_results)
    _save_per_question_csv(config_dir, all_results)

    # ── Save full JSON ──────────────────────────────────────────────────
    full_json = {
        "report": {
            "timestamp": run_ts,
            "benchmark_size": len(benchmark),
            "configurations": len(ABLATION_CONFIGS),
            "elapsed_seconds": round(total_elapsed, 3),
            "ragas_used": use_ragas,
        },
        "configurations": [
            {k: v for k, v in r.items() if k != "questions"} for r in all_results
        ],
    }
    json_path = config_dir / "ablation_full.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, default=str)

    # ── Print report ────────────────────────────────────────────────────
    print(f"\n{report}")

    # ── Final summary ───────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("  Ablation study complete!")
    print(f"  Results saved to: {config_dir}")
    print("    Report:   ablation_report.txt")
    print("    Summary:  ablation_summary.csv")
    print("    Details:  per_question_results.csv")
    print("    Full:     ablation_full.json")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 50}")


def main() -> None:
    """Synchronous entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
