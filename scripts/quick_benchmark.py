#!/usr/bin/env python3
"""
Quick Benchmark: Measure retrieval + generation quality across pipeline configs.
Runs faster than the full ablation study by avoiding the per-query engine setup overhead.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# ── Benchmark questions (subset for speed) ──────────────────────────────────

BENCHMARK = [
    {
        "question": "What is the standard refund window for retail products?",
        "ground_truth": "Retail products are eligible for a full refund within 30 days of purchase with receipt.",
        "category": "Refund & Support",
    },
    {
        "question": "What is the standard office core hours requirement?",
        "ground_truth": "Core collaboration hours are from 10:00 AM to 3:00 PM EST daily.",
        "category": "Corporate Policy",
    },
    {
        "question": "What is the maximum allowed salary for a junior analyst role?",
        "ground_truth": "According to rules.json, the maximum permitted salary for junior tiers is capped at $120,000.",
        "category": "Platform Operations",
    },
    {
        "question": "If an international user returned opened electronics, what is the fee policy?",
        "ground_truth": "International returns require the customer to cover shipping, and opened electronics are subject to a 15% restocking fee.",
        "category": "Multi-hop Reasoning",
    },
]

# ── 3 Pipeline Configs (most impactful) ─────────────────────────────────────

CONFIGS = [
    {
        "id": 1,
        "name": "Dense Only",
        "search_mode": "dense",
        "enable_hyde": False,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 2,
        "name": "Hybrid (RRF)",
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 3,
        "name": "Hybrid + HyDE",
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_reranking": False,
        "top_k": 4,
    },
    {
        "id": 4,
        "name": "Hybrid + Reranking",
        "search_mode": "hybrid",
        "enable_hyde": False,
        "enable_reranking": True,
        "top_k": 5,
    },
    {
        "id": 5,
        "name": "Full CSRAG",
        "search_mode": "hybrid",
        "enable_hyde": True,
        "enable_reranking": True,
        "top_k": 5,
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_contexts(result_state: dict) -> list[str]:
    """Extract context texts from pipeline result."""
    from langchain_core.documents import Document
    contexts = []
    for key in ("good_docs", "docs", "web_docs"):
        docs = result_state.get(key, []) or []
        for d in docs:
            if isinstance(d, Document):
                contexts.append(d.page_content)
            elif isinstance(d, dict):
                contexts.append(d.get("content", d.get("page_content", "")))
    return contexts


# ── Benchmark Runner ───────────────────────────────────────────────────────

async def run_benchmark():
    from app.core.csrag_engine import CSRAGEngine
    from app.core.vector_store import VectorStoreService
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.store.postgres.aio import AsyncPostgresStore
    from app.config import get_settings
    from langchain_core.messages import HumanMessage

    settings = get_settings()
    vector_store = VectorStoreService()

    # Check Qdrant
    info = vector_store.get_collection_info()
    print(f"Qdrant collection: {info['points_count']} points")

    # Connect Postgres
    cm_store = AsyncPostgresStore.from_conn_string(settings.database_url)
    store = await cm_store.__aenter__()
    await store.setup()
    cm_checkpointer = AsyncPostgresSaver.from_conn_string(settings.database_url)
    checkpointer = await cm_checkpointer.__aenter__()
    await checkpointer.setup()

    try:
        engine = CSRAGEngine(vector_store, store, checkpointer)

        all_results = []

        for cfg in CONFIGS:
            print(f"\n{'='*60}")
            print(f"  Config {cfg['id']}: {cfg['name']}")
            print(f"  {cfg['search_mode']}, hyde={cfg['enable_hyde']}, rerank={cfg['enable_reranking']}, top_k={cfg['top_k']}")
            print(f"{'='*60}")

            faith_scores = []
            relev_scores = []
            prec_scores = []

            for q_idx, item in enumerate(BENCHMARK, 1):
                print(f"  [{q_idx}/{len(BENCHMARK)}] {item['question'][:60]}...", end=" ", flush=True)

                t0 = time.monotonic()

                # Build state with forced retrieval + skip memory steps
                init_state = CSRAGEngine._initial_state(
                    question=item["question"],
                    search_mode=cfg["search_mode"],
                    top_k=cfg["top_k"],
                    enable_hyde=cfg["enable_hyde"],
                    enable_reranking=cfg["enable_reranking"],
                )
                # Override: force retrieval by jumping directly to retrieve_docs node
                init_state["need_retrieval"] = True
                init_state["query_type"] = "RAG"
                init_state["retrieval_query"] = item["question"]

                config = {
                    "configurable": {"thread_id": f"bench-cfg{cfg['id']}-q{q_idx}", "user_id": "bench"},
                    "recursion_limit": 80,
                }

                try:
                    result = await engine.run_with_state(init_state, config)
                    elapsed = time.monotonic() - t0

                    answer = result.get("answer", "")
                    contexts = _extract_contexts(result)

                    # Evaluate
                    from app.core.feature3_rag.ragas_evaluator import get_ragas_evaluator
                    evaluator = get_ragas_evaluator()
                    scores = await evaluator.evaluate(
                        question=item["question"],
                        answer=answer,
                        contexts=contexts,
                    )

                    if scores:
                        faith_scores.append(scores.faithfulness)
                        relev_scores.append(scores.answer_relevancy)
                        prec_scores.append(scores.context_precision)
                        print(f"{elapsed:.1f}s faith={scores.faithfulness:.3f}")
                    else:
                        print(f"{elapsed:.1f}s NO SCORES")

                except Exception as e:
                    print(f"FAIL: {type(e).__name__}: {str(e)[:80]}")

            # Aggregate
            n = len(faith_scores) or 1
            metrics = {
                "faithfulness": round(sum(faith_scores) / n, 4) if faith_scores else 0.0,
                "answer_relevancy": round(sum(relev_scores) / n, 4) if relev_scores else 0.0,
                "context_precision": round(sum(prec_scores) / n, 4) if prec_scores else 0.0,
            }
            print(f"\n  >> {cfg['name']}: faith={metrics['faithfulness']:.4f}, relev={metrics['answer_relevancy']:.4f}, prec={metrics['context_precision']:.4f}")

            all_results.append({"config": cfg["name"], "metrics": metrics})

        # Final comparison
        print(f"\n{'='*60}")
        print("  RESULTS SUMMARY")
        print(f"{'='*60}")
        for r in all_results:
            m = r["metrics"]
            print(f"  {r['config']:<20} faith={m['faithfulness']:.4f}  relev={m['answer_relevancy']:.4f}  prec={m['context_precision']:.4f}")

        if len(all_results) >= 2:
            first = all_results[0]["metrics"]
            last = all_results[-1]["metrics"]
            print(f"\n  Improvement ({all_results[0]['config']} -> {all_results[-1]['config']}):")
            for key in ["faithfulness", "answer_relevancy", "context_precision"]:
                delta = last[key] - first[key]
                pct = (delta / first[key] * 100) if first[key] > 0 else 0
                print(f"    {key:<20} {first[key]:.4f} -> {last[key]:.4f}  ({pct:+.1f}%)")

    finally:
        await cm_checkpointer.__aexit__(None, None, None)
        await cm_store.__aexit__(None, None, None)

    print("\n  Done!")


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
