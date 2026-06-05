#!/usr/bin/env python3
"""
Ablation Study — IDOP Pipeline Benchmarking Suite
==================================================

Runs a fixed 50-item benchmark dataset against 5 incremental pipeline
configurations, evaluates using the in-house LLM evaluator, and produces
a timestamped CSV/JSON report.

Pipeline configurations tested:

  1. Dense Only        — Dense vector search only (no hybrid, no HyDE, no reranking)
  2. Hybrid (RRF)      — Dense + sparse hybrid search with Reciprocal Rank Fusion
  3. Hybrid + HyDE     — Hybrid search + Hypothetical Document Embeddings query expansion
  4. Hybrid + Rerank   — Hybrid search + Voyage AI cross-encoder reranking
  5. Full CSRAG        — All features: HyDE + hybrid + reranking + CRAG + SRAG

Usage:
    # Default — run all 50 questions through all 5 configs
    python scripts/eval_ragas.py

    # Run only a subset (faster for iterative development)
    python scripts/eval_ragas.py --subset 10

    # Run one slice (to stay within API rate limits):
    python scripts/eval_ragas.py --slice 1/5 --config 1  # 10 Qs, Dense Only
    python scripts/eval_ragas.py --slice 2/5 --config 2  # 10 Qs, Hybrid (RRF)
    python scripts/eval_ragas.py --slice 3/5 --config 3  # 10 Qs, Hybrid + HyDE
    python scripts/eval_ragas.py --slice 4/5 --config 4  # 10 Qs, Hybrid + Rerank
    python scripts/eval_ragas.py --slice 5/5 --config 5  # 10 Qs, Full CSRAG

    # Run one config on all 50 questions:
    python scripts/eval_ragas.py --config 1

    # Combine all 5 slices into a single report:
    python scripts/eval_ragas.py --combine data/ablation_results/run_<timestamp>

    # Skip pipeline execution (re-evaluate last run's results)
    python scripts/eval_ragas.py --reuse-last

Output:
    data/ablation_results/
    └── run_<timestamp>/
        ├── config_<id>_<name>.json     # Per-configuration results
        ├── ablation_summary.csv        # Summary table per configuration
        ├── ablation_report.txt         # Human-readable report
        ├── per_question_results.csv    # Per-question detailed results
        └── ablation_full.json          # All results combined
"""

# ═══════════════════════════════════════════════════════════════════════════════
# Imports
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Dataset — 50 questions across 5 categories
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK = [
    # ── Category 1: Version Conflicts (10 questions) ─────────────────────
    {
        "question": "What is the standard refund window for retail products?",
        "ground_truth": "Retail products are eligible for a full refund within 30 days of purchase with receipt.",
        "category": "Version Conflict",
    },
    {
        "question": "Are there any restocking fees for opened electronics?",
        "ground_truth": "Opened electronics are subject to a 15% restocking fee if returned within the 30-day window.",
        "category": "Version Conflict",
    },
    {
        "question": "What is the international return shipping policy for customers?",
        "ground_truth": "For international orders, customers must cover return shipping costs unless the product arrived damaged.",
        "category": "Version Conflict",
    },
    {
        "question": "Can clearance items be returned for a refund?",
        "ground_truth": "All clearance items are marked as final sale and are strictly non-refundable.",
        "category": "Version Conflict",
    },
    {
        "question": "What is the employee referral bonus amount and how is it paid?",
        "ground_truth": "The standard referral bonus is $2,000, paid in two installments after the new hire completes 90 and 180 days.",
        "category": "Version Conflict",
    },
    {
        "question": "How many weeks of parental leave are available for new parents?",
        "ground_truth": "The platform provides 12 weeks of fully paid parental leave for primary and secondary caregivers after 1 year of service.",
        "category": "Version Conflict",
    },
    {
        "question": "Are contract workers eligible for dental insurance benefits?",
        "ground_truth": "Contract workers are generally ineligible for corporate group dental benefits unless explicitly detailed in their agreement.",
        "category": "Version Conflict",
    },
    {
        "question": "What is the annual tuition reimbursement cap for employees?",
        "ground_truth": "The company reimburses up to $5,250 annually for pre-approved, job-related graduate coursework with grade B or higher.",
        "category": "Version Conflict",
    },
    {
        "question": "How many consecutive sick days require a doctor's note?",
        "ground_truth": "Consecutive sick leaves of 3 or more days require a valid medical certificate submitted to HR.",
        "category": "Version Conflict",
    },
    {
        "question": "Can customers exchange gift cards for cash or store credit?",
        "ground_truth": "Gift cards cannot be returned, exchanged, or redeemed for cash refunds under any circumstances.",
        "category": "Version Conflict",
    },
    # ── Category 2: Out-of-Document Knowledge (10 questions) ────────────
    {
        "question": "What is the refund policy for the new subscription box service?",
        "ground_truth": "Customers may cancel within 14 days of the first billing date for a full refund; subsequent months are non-refundable but can be cancelled to stop future billing.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "How are refund disputes handled for items sold by third-party marketplace sellers?",
        "ground_truth": "The refund policy is managed by the seller. If the seller is unresponsive for more than 7 days, IDOP may issue a courtesy credit of up to 50% of the purchase price.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What is the refund policy for IDOP Collectibles novelty items?",
        "ground_truth": "Novelty items including collectible pins, limited-edition packaging, and event merchandise are final sale and non-refundable.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "Are novelty items non-refundable for EU customers too?",
        "ground_truth": "Yes, novelty items are final sale and non-refundable for ALL markets including EU customers.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What benefits do contract workers lose in the 2026 policy transition?",
        "ground_truth": "Contract workers lose access to dental benefits and the 401(k) match as of February 1, 2026.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What is the refund processing time for customers in Latin America?",
        "ground_truth": "Refunds to LATAM customers may take up to 15 business days to process due to local banking regulations.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What is the mandatory legal warranty period for products sold in the European Union?",
        "ground_truth": "All products sold to EU customers carry a mandatory 2-year legal warranty covering manufacturing defects.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What employee discount is available on company products?",
        "ground_truth": "No employee discount policy exists in the current documentation.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What is the data deletion policy for EU customers under GDPR?",
        "ground_truth": "EU customers have the right to request complete deletion of their personal data under GDPR Article 17. Requests must be processed within 30 calendar days.",
        "category": "Out-of-Document Knowledge",
    },
    {
        "question": "What is the company's policy on cryptocurrency refunds?",
        "ground_truth": "No cryptocurrency refund policy exists in the current documentation.",
        "category": "Out-of-Document Knowledge",
    },
    # ── Category 3: Regional & Policy Variations (10 questions) ─────────
    {
        "question": "What return window do APAC customers have for electronics?",
        "ground_truth": "APAC customers have a 7-day return window for electronics under local regulations.",
        "category": "Regional Policy",
    },
    {
        "question": "What restocking fee applies to EU customers returning opened electronics?",
        "ground_truth": "No restocking fees may be applied to EU customer returns under EU consumer protection laws.",
        "category": "Regional Policy",
    },
    {
        "question": "What currency do LATAM customers receive refunds in?",
        "ground_truth": "Refunds to LATAM customers are processed in local currency at the exchange rate on the date of refund approval.",
        "category": "Regional Policy",
    },
    {
        "question": "Which APAC markets receive free return shipping?",
        "ground_truth": "Customers in Japan, South Korea, and Singapore receive free return shipping.",
        "category": "Regional Policy",
    },
    {
        "question": "Do EU customers have to pay for return shipping on international orders?",
        "ground_truth": "No, the company bears all return shipping costs for EU customers within the return window.",
        "category": "Regional Policy",
    },
    {
        "question": "What is the standard APAC return window for non-electronics?",
        "ground_truth": "APAC customers have a 14-day return window for all products except electronics.",
        "category": "Regional Policy",
    },
    {
        "question": "What happens if an EU customer requests deletion of their personal data?",
        "ground_truth": "EU data deletion requests must be processed within 30 calendar days under GDPR Article 17.",
        "category": "Regional Policy",
    },
    {
        "question": "What is the APAC restocking fee percentage on opened electronics?",
        "ground_truth": "A 20% restocking fee applies to opened electronics in APAC markets.",
        "category": "Regional Policy",
    },
    {
        "question": "Can LATAM customers receive refunds in US dollars?",
        "ground_truth": "No, refunds to LATAM customers are processed in local currency at the prevailing exchange rate.",
        "category": "Regional Policy",
    },
    {
        "question": "What warranty do EU customers have in addition to the standard refund policy?",
        "ground_truth": "EU customers have a mandatory 2-year legal warranty covering manufacturing defects.",
        "category": "Regional Policy",
    },
    # ── Category 4: Multi-hop Synthesis (10 questions) ──────────────────
    {
        "question": "If an EU customer returns opened electronics bought in January 2026 before the new policy took effect, what fees apply?",
        "ground_truth": "No restocking fee applies because the order was under the 2025 policy (no restocking fee for electronics) and EU policies override standard fees.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "An employee hired in 2023 wants to know their tuition reimbursement cap under the current policy. What is it?",
        "ground_truth": "Employees hired before January 1, 2024 are grandfathered with a $6,000 annual tuition reimbursement cap.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "A customer bought a clearance item in January 2026 but is returning it in February 2026. What refund policy applies?",
        "ground_truth": "Orders before February 1, 2026 are governed by the 2025 policy. Clearance could be returned with 25% restocking fee under 2025 policy.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "An employee hired in 2023 wants parental leave. How many weeks and at what pay?",
        "ground_truth": "Employees hired before January 1, 2024 are grandfathered with 14 weeks of fully paid parental leave.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "A customer bought a subscription box service and wants to cancel after 3 months. What refund are they entitled to?",
        "ground_truth": "Customers may cancel within 14 days of the first billing date for a full refund. Subsequent months are non-refundable but can be cancelled.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "What is the combined net effect of the 2026 benefit changes on an employee's total compensation package?",
        "ground_truth": "Wellness +$200, tuition +$2,250, referral +$500, parental leave +4 weeks. Only contract worker dental/401(k) were reduced.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "A contractor hired in 2025 submits a dental insurance claim in March 2026. Is it covered?",
        "ground_truth": "No. Contract worker dental benefits were eliminated effective February 1, 2026.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "Can an EU customer return an IDOP Collectibles novelty item?",
        "ground_truth": "No. Novelty items are final sale for ALL markets including EU, overriding standard EU return policies.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "An employee referred a new hire who starts March 1 and resigns after 100 days. What referral bonus is paid?",
        "ground_truth": "Only the first $1,000 installment is paid at 90 days. The second $1,000 at 180 days is voided.",
        "category": "Multi-hop Synthesis",
    },
    {
        "question": "A customer bought opened electronics internationally from Japan. What fees and shipping costs apply?",
        "ground_truth": "20% restocking fee (APAC rate) and free return shipping (Japan is one of the free-shipping APAC markets).",
        "category": "Multi-hop Synthesis",
    },
    # ── Category 5: Ambiguous Queries & Edge Cases (10 questions) ───────
    {
        "question": "What changed in the employee benefits this year compared to last year?",
        "ground_truth": "Wellness +$200, tuition +$2,250, parental leave +4 weeks, referral +$500. Contract worker benefits eliminated.",
        "category": "Ambiguous Query",
    },
    {
        "question": "What is the return policy update from last year?",
        "ground_truth": "Return window reduced 45 to 30 days, international shipping shifted to customer, 15% restocking fee introduced, clearance became non-refundable.",
        "category": "Ambiguous Query",
    },
    {
        "question": "Can I get my money back if I change my mind after buying something?",
        "ground_truth": "Standard retail: 30-day refund. Clearance: final sale. Electronics: 15% restocking fee. Gift cards: no. Bundles: must return together.",
        "category": "Ambiguous Query",
    },
    {
        "question": "What benefits did the company cut this year?",
        "ground_truth": "Only contract worker dental and 401(k) match were eliminated. All other benefits increased despite the budget cuts label.",
        "category": "Ambiguous Query",
    },
    {
        "question": "Do I need a doctor's note if I call in sick?",
        "ground_truth": "1-2 days: no. 3+ consecutive days: medical certificate required.",
        "category": "Ambiguous Query",
    },
    {
        "question": "What is the best way to return an expensive item I just bought?",
        "ground_truth": "Depends on item type, region, and purchase date. Electronics may incur restocking fees. Regional policies may override.",
        "category": "Ambiguous Query",
    },
    {
        "question": "Can executives and VPs fly business class on business trips?",
        "ground_truth": "Flights under 6 hours: economy class regardless of seniority. Flights over 6 hours: no restrictions.",
        "category": "Ambiguous Query",
    },
    {
        "question": "What happens to my tuition reimbursement if I get a grade of C in a course?",
        "ground_truth": "Under 2026 policy: grade C does not qualify (minimum B). Under superseded 2025 policy: grade C qualified up to $3,000 cap.",
        "category": "Ambiguous Query",
    },
    {
        "question": "I am an EU customer and bought a laptop that arrived with a manufacturing defect. What are my rights?",
        "ground_truth": "2-year legal warranty covers defects. Can return within 30 days at no cost with no restocking fee. Company pays return shipping.",
        "category": "Ambiguous Query",
    },
    {
        "question": "What is the total monetary value of all employee benefit increases from 2025 to 2026?",
        "ground_truth": "$2,950 total: wellness +$200, tuition +$2,250, referral +$500.",
        "category": "Ambiguous Query",
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configurations
# ═══════════════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "name": "Dense Only",
        "description": "Baseline - single dense vector similarity search",
        "search_mode": "dense",
        "enable_hyde": False,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 2,
        "name": "Hybrid (RRF)",
        "description": "Dense + sparse hybrid search with RRF fusion",
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 3,
        "name": "Hybrid + HyDE",
        "description": "Hybrid search + HyDE query expansion",
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 4,
        "name": "Hybrid + Reranking",
        "description": "Hybrid search + Voyage AI reranking",
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_reranking": True,
        "top_k": 5,
    },
    {
        "id": 5,
        "name": "Full CSRAG",
        "description": "HyDE + hybrid + reranking + CRAG + SRAG + context enrichment",
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
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / total)
    return f"[{'#' * filled}{'.' * (width - filled)}] {current}/{total}"


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════════════════════════════════════


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


async def _create_engine() -> tuple:
    """Create a shared CSRAGEngine with Postgres resources."""
    from app.core.csrag_engine import CSRAGEngine
    from app.core.vector_store import VectorStoreService
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from app.config import get_settings

    settings = get_settings()
    vector_store = VectorStoreService()

    cm_store = AsyncPostgresStore.from_conn_string(settings.database_url)
    store = await cm_store.__aenter__()
    try:
        await store.setup()
    except BaseException:
        await cm_store.__aexit__(None, None, None)
        raise

    cm_checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
    checkpointer = await cm_checkpointer.__aenter__()
    try:
        await checkpointer.setup()
    except BaseException:
        await cm_checkpointer.__aexit__(None, None, None)
        await cm_store.__aexit__(None, None, None)
        raise

    engine = CSRAGEngine(vector_store, store, checkpointer)
    return engine, store, checkpointer, cm_store, cm_checkpointer


async def run_single_query(
    question: str,
    config: Dict[str, Any],
    config_dir: Path,
    engine: Any,
) -> Dict[str, Any]:
    """Run one question through one pipeline configuration."""
    from app.core.csrag_engine import CSRAGEngine

    thread_id = f"ablation-cfg{config['id']}-{config_dir.name}"
    graph_config = {
        "configurable": {"thread_id": thread_id, "user_id": "ablation-eval"},
        "recursion_limit": 80,
    }

    init_state = CSRAGEngine._initial_state(
        question=question,
        search_mode=config["search_mode"],
        top_k=config["top_k"],
        enable_hyde=config["enable_hyde"],
        enable_reranking=config["enable_reranking"],
        enable_ragas=False,
    )
    init_state["need_retrieval"] = True
    init_state["query_type"] = "RAG"
    init_state["retrieval_query"] = question
    init_state["user_id"] = "ablation-eval"

    t0 = time.monotonic()
    result_state = await engine.run_with_state(init_state, graph_config)
    elapsed = time.monotonic() - t0

    answer = result_state.get("answer", "")
    contexts = _collect_contexts(result_state)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "pipeline_time_s": round(elapsed, 3),
        "crag_verdict": result_state.get("crag_verdict", ""),
        "issup": result_state.get("issup", ""),
        "isuse": result_state.get("isuse", ""),
    }


async def run_configuration(
    config: Dict[str, Any],
    benchmark: List[Dict[str, Any]],
    config_dir: Path,
) -> Dict[str, Any]:
    """Run pipeline queries (no evaluation). Returns raw pipeline outputs."""
    cfg_id = config["id"]
    cfg_name = config["name"]
    n_questions = len(benchmark)

    print(f"\n{'=' * 70}")
    print(f"  Run {cfg_id}/5: {cfg_name}")
    print(f"  {config['description']}")
    print(
        f"  search_mode={config['search_mode']}, hyde={config['enable_hyde']}, rerank={config['enable_reranking']}, top_k={config['top_k']}"
    )
    print(f"{'=' * 70}")

    print("  [INIT] Creating CSRAGEngine...", end="", flush=True)
    engine, store, checkpointer, cm_store, cm_checkpointer = await _create_engine()
    print(" OK")

    pipeline_outputs: List[Dict[str, Any]] = []
    start_time = time.monotonic()

    try:
        for q_idx, item in enumerate(benchmark, start=1):
            progress = _progress_bar(q_idx, n_questions)
            short_q = item["question"][:70]
            print(f"  {progress}  {short_q}...", end="", flush=True)

            try:
                out = await run_single_query(
                    item["question"], config, config_dir, engine
                )
                out["ground_truth"] = item["ground_truth"]
                out["category"] = item.get("category", "")
                out["config_id"] = cfg_id
                pipeline_outputs.append(out)
                print(f"  OK ({out['pipeline_time_s']:.1f}s)")
                await asyncio.sleep(
                    5.0
                )  # Rate-limit delay between questions (5s to stay under 30 RPM)
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
    finally:
        await cm_checkpointer.__aexit__(None, None, None)
        await cm_store.__aexit__(None, None, None)

    total_time = time.monotonic() - start_time
    avg_time = round(total_time / n_questions, 3)

    return {
        "config_id": cfg_id,
        "name": cfg_name,
        "description": config["description"],
        "parameters": {
            k: config[k]
            for k in ("search_mode", "enable_hyde", "enable_reranking", "top_k")
        },
        "timing": {
            "total_seconds": round(total_time, 3),
            "avg_seconds_per_query": avg_time,
        },
        "questions": pipeline_outputs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation (in-house only - rate-limit-safe)
# ═══════════════════════════════════════════════════════════════════════════════


async def evaluate_config(
    pipeline_outputs: List[Dict[str, Any]],
    config_id: int,
    config_name: str,
) -> Dict[str, float]:
    """
    Evaluate one config's pipeline outputs using the in-house evaluator.
    Adds delays between LLM calls to respect rate limits.

    Each question makes 3 LLM calls (relevancy, faithfulness, precision).
    With 50 questions x 3 calls = 150 calls per config.
    """
    print(
        f"\n  [EVAL] Evaluating {config_name} ({len(pipeline_outputs)} questions)...",
        flush=True,
    )
    from app.core.feature3_rag.ragas_evaluator import get_ragas_evaluator

    evaluator = get_ragas_evaluator()
    faith_scores: List[float] = []
    relev_scores: List[float] = []
    prec_scores: List[float] = []
    recall_numer: List[float] = []
    eval_errors = 0

    for q_idx, out in enumerate(pipeline_outputs, start=1):
        answer = out.get("answer", "")
        contexts = out.get("contexts", [])

        if not answer or answer.startswith("[ERROR]"):
            print(f"    [{q_idx}] SKIP (error or empty)", flush=True)
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
                total = max(scores.context_total_count, 1)
                recall_numer.append(scores.context_relevant_count / total)
                print(
                    f"    [{q_idx}] f={scores.faithfulness:.3f} r={scores.answer_relevancy:.3f} p={scores.context_precision:.3f}",
                    flush=True,
                )
            else:
                eval_errors += 1
                print(f"    [{q_idx}] NO SCORES returned", flush=True)
        except Exception as e:
            eval_errors += 1
            print(f"    [{q_idx}] ERROR: {type(e).__name__}", flush=True)

        # Rate-limit delay between evaluation questions
        await asyncio.sleep(3.0)

    n = len(faith_scores) or 1
    metrics = {
        "faithfulness": round(sum(faith_scores) / n, 4) if faith_scores else 0.0,
        "answer_relevancy": round(sum(relev_scores) / n, 4) if relev_scores else 0.0,
        "context_precision": round(sum(prec_scores) / n, 4) if prec_scores else 0.0,
        "context_recall": round(sum(recall_numer) / n, 4) if recall_numer else 0.0,
    }
    print(
        f"  [EVAL] {config_name} complete - "
        f"faith={metrics['faithfulness']:.3f} relev={metrics['answer_relevancy']:.3f} "
        f"prec={metrics['context_precision']:.3f} recall={metrics['context_recall']:.3f} "
        f"({len(faith_scores)} evaluated, {eval_errors} errors)",
        flush=True,
    )
    return metrics


def _compute_category_metrics(
    outputs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-category success rates."""
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

    return {
        cat: {
            "total": len(results),
            "successful": sum(results),
            "success_rate": round(sum(results) / len(results), 4) if results else 0.0,
        }
        for cat, results in categories.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════


def _build_report(
    all_results: List[Dict[str, Any]],
    benchmark_name: str,
    total_elapsed: float,
    actual_questions: int = 50,
) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("  IDOP ABLATION STUDY REPORT")
    lines.append(f"  Benchmark: {benchmark_name} ({actual_questions} questions)")
    lines.append(f"  Completed: {_timestamp()}")
    lines.append(f"  Total time: {total_elapsed:.1f}s")
    lines.append("=" * 72)
    lines.append("")

    header = f"{'Run':<5} {'Configuration':<30} {'Faith':<9} {'Relev':<9} {'Prec':<9} {'Recall':<9} {'Avg Q(s)':<9}"
    lines.append(header)
    lines.append("-" * len(header))

    base_metrics = None
    for r in all_results:
        m = r["metrics"]
        imp = (
            f" ({m['faithfulness'] - base_metrics['faithfulness']:+.1%})"
            if base_metrics and base_metrics.get("faithfulness", 0) > 0
            else ""
        )
        lines.append(
            f"{r['config_id']:<5} {r['name']:<30} {m['faithfulness']:<9.4f}{imp:<9} {m['answer_relevancy']:<9.4f} {m['context_precision']:<9.4f} {m['context_recall']:<9.4f} {r['timing']['avg_seconds_per_query']:<9.3f}"
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

    for r in all_results:
        lines.append(f"\n{'-' * 50}")
        lines.append(f"  Configuration {r['config_id']}: {r['name']}")
        lines.append(f"  {r['description']}")
        lines.append(f"  Parameters: {json.dumps(r['parameters'])}")
        lines.append(f"  Avg query time: {r['timing']['avg_seconds_per_query']:.3f}s")
        m = r["metrics"]
        lines.append(f"    faithfulness       = {m['faithfulness']:.4f}")
        lines.append(f"    answer_relevancy    = {m['answer_relevancy']:.4f}")
        lines.append(f"    context_precision   = {m['context_precision']:.4f}")
        lines.append(f"    context_recall      = {m['context_recall']:.4f}")
        if r.get("category_metrics"):
            lines.append("    Per-Category Success Rates:")
            for cat, cm in sorted(r["category_metrics"].items()):
                lines.append(
                    f"      {cat:<25} {cm['successful']:>3}/{cm['total']:<3} ({cm['success_rate']:.1%})"
                )

    lines.append("")
    lines.append("=" * 72)
    lines.append("  END OF REPORT")
    lines.append("=" * 72)
    return "\n".join(lines)


def _save_csv(results_dir: Path, all_results: List[Dict[str, Any]]) -> Path:
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
        description="IDOP Ablation Study - benchmark 5 pipeline configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset",
        "-n",
        type=int,
        default=None,
        help="Run on a random subset of N questions",
    )
    parser.add_argument(
        "--slice",
        type=str,
        default=None,
        help="Run slice N/M (e.g., 1/5 = questions 0-9, 2/5 = 10-19)",
    )
    parser.add_argument(
        "--combine",
        type=str,
        default=None,
        help="Combine all slices from a run directory into a single report",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/ablation_results",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        type=int,
        default=None,
        help="Run only a specific config (1-5). Default: all 5 configs",
    )
    parser.add_argument(
        "--reuse-last",
        action="store_true",
        help="Skip pipeline execution, re-evaluate last run's results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subset selection"
    )
    return parser.parse_args(argv)


def _parse_slice(slice_str: str) -> tuple[int, int]:
    """Parse '1/5' into (slice_num=0, total_slices=5)."""
    parts = slice_str.split("/")
    if len(parts) != 2:
        print(f"  [ERROR] Invalid --slice format: '{slice_str}'. Use N/M (e.g., 1/5)")
        sys.exit(1)
    try:
        num = int(parts[0]) - 1  # 0-indexed
        total = int(parts[1])
        if num < 0 or num >= total:
            raise ValueError
        return num, total
    except ValueError:
        print(f"  [ERROR] Invalid --slice values: '{slice_str}'. N must be 1..M")
        sys.exit(1)


def _get_slice(num: int, total: int, full: list) -> list:
    """Get a contiguous slice of the benchmark.

    Slices are evenly divided. The last slice gets any remainder.
    """
    n = len(full)
    slice_size = n // total
    start = num * slice_size
    if num == total - 1:
        end = n  # last slice gets remainder
    else:
        end = start + slice_size
    return full[start:end]


async def main_async() -> None:
    args = parse_args()
    benchmark = list(BENCHMARK)

    # ── Handle --combine ─────────────────────────────────────────────────
    if args.combine:
        combine_dir = Path(args.combine)
        if not combine_dir.is_dir():
            print(f"  [ERROR] Combine directory not found: {combine_dir}")
            sys.exit(1)
        print(f"\n  [COMBINE] Merging slice results from {combine_dir}...")
        # Find all slice subdirectories
        slice_dirs = sorted(combine_dir.glob("slice_*"))
        if not slice_dirs:
            print(f"  [ERROR] No slice_* directories found in {combine_dir}")
            sys.exit(1)
        print(
            f"  [COMBINE] Found {len(slice_dirs)} slices: {[d.name for d in slice_dirs]}"
        )

        all_results = []
        total_questions = 0
        total_pipeline_time = 0.0

        for sd in slice_dirs:
            for cfg in ABLATION_CONFIGS:
                cfg_file = (
                    sd
                    / f"config_{cfg['id']:02d}_{cfg['name'].replace(' ', '_').lower()}.json"
                )
                if cfg_file.exists():
                    with open(cfg_file) as f:
                        data = json.load(f)
                    # Find or create the config-level result
                    existing = next(
                        (r for r in all_results if r["config_id"] == data["config_id"]),
                        None,
                    )
                    if existing:
                        # Merge metrics (weighted average)
                        old_n = existing.get("_question_count", 0)
                        new_n = data.get("question_count", 10)
                        total_n = old_n + new_n
                        for key in (
                            "faithfulness",
                            "answer_relevancy",
                            "context_precision",
                            "context_recall",
                        ):
                            existing["metrics"][key] = (
                                existing["metrics"][key] * old_n
                                + data["metrics"][key] * new_n
                            ) / total_n
                        existing["_question_count"] = total_n
                        existing["timing"]["total_seconds"] += data["timing"][
                            "total_seconds"
                        ]
                        existing["timing"]["avg_seconds_per_query"] = round(
                            existing["timing"]["total_seconds"] / total_n, 3
                        )
                    else:
                        data["_question_count"] = data.get("question_count", 10)
                        all_results.append(data)

        all_results.sort(key=lambda r: r["config_id"])
        total_pipeline_time = sum(r["timing"]["total_seconds"] for r in all_results)

        # Generate combined report
        total_questions = sum(r.get("_question_count", 0) for r in all_results)
        report = _build_report(
            all_results,
            f"IDOP Combined ({total_questions} questions)",
            total_pipeline_time,
        )
        report_path = combine_dir / "ablation_report_combined.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        _save_csv(combine_dir, all_results)

        full_json = {
            "report": {
                "timestamp": _timestamp(),
                "benchmark_size": total_questions,
                "configurations": len(ABLATION_CONFIGS),
                "combined_from": len(slice_dirs),
                "elapsed_seconds": round(total_pipeline_time, 3),
            },
            "configurations": [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("questions", "_question_count")
                }
                for r in all_results
            ],
        }
        with open(combine_dir / "ablation_combined.json", "w", encoding="utf-8") as f:
            json.dump(full_json, f, indent=2, default=str)

        print(f"\n{report}")
        print(f"\n  [COMBINE] Report saved to {report_path}")
        return

    # ── Handle --slice (subset of questions by index) ────────────────────
    if args.slice:
        slice_num, total_slices = _parse_slice(args.slice)
        benchmark = _get_slice(slice_num, total_slices, benchmark)
        print(
            f"  [SLICE] Using slice {args.slice} ({len(benchmark)} questions: idx {slice_num * (50 // total_slices)}-{slice_num * (50 // total_slices) + len(benchmark) - 1})"
        )

    # ── Handle --subset (random subset, applied AFTER --slice) ───────────
    if args.subset and 0 < args.subset < len(benchmark):
        rng = random.Random(args.seed)
        rng.shuffle(benchmark)
        benchmark = benchmark[: args.subset]
        print(
            f"  [SUBSET] Using {len(benchmark)} questions (--subset={args.subset}, seed={args.seed})"
        )

    if not args.slice and not args.subset:
        print(f"  [FULL] Using all {len(benchmark)} benchmark questions")

    # Filter configs if --config is specified
    configs_to_run = list(ABLATION_CONFIGS)
    if args.config is not None:
        if args.config < 1 or args.config > len(ABLATION_CONFIGS):
            print(
                f"  [ERROR] Invalid --config: {args.config}. Must be 1-{len(ABLATION_CONFIGS)}"
            )
            sys.exit(1)
        configs_to_run = [cfg for cfg in ABLATION_CONFIGS if cfg["id"] == args.config]
        print(
            f"  [CONFIG] Running only config {args.config}: {configs_to_run[0]['name']}"
        )

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_ts = _timestamp()
    if args.slice:
        slice_label = args.slice.replace("/", "-")
        config_dir = base_dir / f"run_{run_ts}_slice{slice_label}"
    else:
        config_dir = base_dir / f"run_{run_ts}"
    # Also create a parent 'combined' directory for later --combine
    # All slices for the same run will be under run_<ts>/slice_*/
    if args.slice:
        # Use parent dir for combining
        parent_dir = base_dir / f"run_{run_ts}"
        parent_dir.mkdir(parents=True, exist_ok=True)
        config_dir = parent_dir / f"slice_{slice_label}"
    config_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Run pipeline (or reuse last) ───────────────────────────
    if args.reuse_last:
        print("\n  [REUSE] Loading last run's pipeline outputs...")
        runs = sorted(base_dir.glob("run_*"), reverse=True)
        if len(runs) < 2:
            print("  [ERROR] No previous run found")
            sys.exit(1)
        prev = runs[1]
        print(f"  [REUSE] Loading from {prev.name}")
        all_pipeline_results = []
        for cfg in configs_to_run:
            cfg_file = (
                prev
                / f"config_{cfg['id']:02d}_{cfg['name'].replace(' ', '_').lower()}.json"
            )
            if cfg_file.exists():
                with open(cfg_file) as f:
                    all_pipeline_results.append(json.load(f))
                print(f"    Loaded {cfg_file.name}")
            else:
                print(f"    [WARN] {cfg_file.name} not found")
        total_pipeline_elapsed = 0.0
    else:
        print(f"\n  {'=' * 50}")
        print(f"  Phase 1: Running pipeline for {len(ABLATION_CONFIGS)} configurations")
        print(f"  Output: {config_dir}")
        print(f"  {'=' * 50}")

        all_pipeline_results = []
        t_start = time.monotonic()
        for cfg in configs_to_run:
            result = await run_configuration(cfg, benchmark, config_dir)
            all_pipeline_results.append(result)
        total_pipeline_elapsed = time.monotonic() - t_start

        # Save raw pipeline outputs immediately (before evaluation)
        for r in all_pipeline_results:
            cfg_file = (
                config_dir
                / f"config_{r['config_id']:02d}_{r['name'].replace(' ', '_').lower()}.json"
            )
            with open(cfg_file, "w", encoding="utf-8") as f:
                summary = {k: v for k, v in r.items() if k != "questions"}
                summary["question_count"] = len(r.get("questions", []))
                json.dump(summary, f, indent=2, default=str)
        print(f"\n  [PHASE 1] Pipeline complete - {total_pipeline_elapsed:.1f}s")

    # ── Phase 2: Evaluate each config separately with rate-limit delays ─
    print(f"\n  {'=' * 50}")
    print(f"  Phase 2: Evaluating {len(all_pipeline_results)} configurations")
    print(
        "  (5s delay between pipeline questions, 3s between eval questions, 10s between configs)"
    )
    print(f"  {'=' * 50}")

    for idx, r in enumerate(all_pipeline_results):
        cfg_id = r["config_id"]
        cfg_name = r["name"]
        questions = r.get("questions", [])
        total_cfgs = len(all_pipeline_results)
        print(
            f"\n  >>> Evaluating Config {idx+1}/{total_cfgs}: {cfg_name} ({len(questions)} questions)",
            flush=True,
        )

        metrics = await evaluate_config(questions, cfg_id, cfg_name)
        r["metrics"] = metrics
        r["category_metrics"] = _compute_category_metrics(questions)

        # Save intermediate results (with metrics now included)
        cfg_file = (
            config_dir
            / f"config_{cfg_id:02d}_{cfg_name.replace(' ', '_').lower()}.json"
        )
        with open(cfg_file, "w", encoding="utf-8") as f:
            summary = {k: v for k, v in r.items() if k != "questions"}
            summary["question_count"] = len(questions)
            json.dump(summary, f, indent=2, default=str)
        print(f"    [SAVED] {cfg_file.name}", flush=True)

        # 10s delay between configs to allow rate limits to reset
        if idx < len(all_pipeline_results) - 1:
            print("    Waiting 10s before next config...", flush=True)
            await asyncio.sleep(10)

    total_elapsed = (
        total_pipeline_elapsed
        + sum(len(r.get("questions", [])) * 2.0 for r in all_pipeline_results)
        + (len(all_pipeline_results) - 1) * 10.0
    )
    # Build final report
    n_questions_actual = len(benchmark)
    report = _build_report(
        all_pipeline_results,
        f"IDOP Benchmark ({n_questions_actual} questions)",
        total_elapsed,
        actual_questions=n_questions_actual,
    )
    report_path = config_dir / "ablation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    _save_csv(config_dir, all_pipeline_results)
    _save_per_question_csv(config_dir, all_pipeline_results)

    full_json = {
        "report": {
            "timestamp": run_ts,
            "benchmark_size": len(benchmark),
            "configurations": len(all_pipeline_results),
            "elapsed_seconds": round(total_elapsed, 3),
        },
        "configurations": [
            {k: v for k, v in r.items() if k != "questions"}
            for r in all_pipeline_results
        ],
    }
    with open(config_dir / "ablation_full.json", "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2, default=str)

    print(f"\n{report}")
    print(f"\n{'=' * 50}")
    print("  Study complete!")
    print(f"  Results saved to: {config_dir}")
    print(f"{'=' * 50}")


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
