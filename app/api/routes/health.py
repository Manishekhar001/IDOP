import asyncio
import sys
from datetime import UTC, datetime
from typing import Any

import psycopg2
from fastapi import APIRouter, Request, status

from app.api.schemas import (
    DetailedHealthResponse,
    DetailedReadinessResponse,
    SystemInfoResponse,
    SystemStatsResponse,
)
from app.config import get_settings
from app.opik import track
from app.services.cache_init import get_doc_cache, get_query_cache

router = APIRouter(tags=["System Diagnostics"])


def _format_redis_cache(query_cache) -> dict:
    """Format query cache stats into RedisCacheStatus-compatible dict."""
    if not query_cache or (not query_cache.enabled and not query_cache.use_local):
        return {
            "status": "disabled",
            "message": "Redis not connected and local fallback not active",
        }

    mode = "redis" if query_cache.enabled else "local_fallback"
    stats = query_cache.get_stats() if hasattr(query_cache, "get_stats") else {}
    cache_types = stats.get("cache_types", {})

    total_hits = sum(ct.get("hits", 0) for ct in cache_types.values())
    total_queries = sum(ct.get("total_queries", 0) for ct in cache_types.values())
    hit_rate = f"{(total_hits / max(total_queries, 1)) * 100:.1f}%"

    # Estimate cost savings (same logic as /stats route)
    cost_estimates = {
        "rag": 0.05,
        "embedding": 0.0001,
        "sql_gen": 0.08,
        "sql_result": 0.01,
    }
    total_savings = sum(
        ct.get("hits", 0) * cost_estimates.get(ct_name, 0)
        for ct_name, ct in cache_types.items()
    )

    return {
        "status": mode,
        "message": (
            "Redis cache connected and operational"
            if mode == "redis"
            else "Redis not connected — using local in-memory fallback"
        ),
        "hit_rate": hit_rate,
        "total_savings": f"${total_savings:.4f}",
    }


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    response_model=DetailedHealthResponse,
    summary="Enhanced Health Check — detailed per-service status, feature flags, config status, Qdrant info, and Redis metrics",
)
@track(name="health_check")
async def health_check(request: Request) -> dict[str, Any]:
    """
    Enhanced Health check endpoint to verify the API is running and check service connectivity.
    Queries Qdrant, PostgreSQL, Upstash Redis, S3/local backends, and external LLM/Search provider keys.
    """
    settings = get_settings()

    # 1. Check Qdrant Connection
    qdrant_connected = False
    collection_info = {}
    try:
        vector_store = request.app.state.vector_store
        qdrant_connected = vector_store.health_check()
        collection_info = vector_store.get_collection_info()
    except Exception:
        pass

    # 2. Check Postgres Connection (checkpointer) — off the event loop
    postgres_connected = False
    try:

        def _check_postgres():
            conn = psycopg2.connect(settings.database_url)
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            conn.close()
            return True

        postgres_connected = await asyncio.to_thread(_check_postgres)
    except Exception:
        pass

    # Check Supabase Connection (company data) — off the event loop
    supabase_connected = False
    if settings.supabase_db_url:
        try:

            def _check_supabase():
                conn = psycopg2.connect(settings.supabase_db_url)
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                conn.close()
                return True

            supabase_connected = await asyncio.to_thread(_check_supabase)
        except Exception:
            pass

    # 3. Check cache availability and actual runtime backend types
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()
    query_cache_status = query_cache.enabled if query_cache else False
    query_cache_mode = (
        "redis"
        if query_cache_status
        else (
            "local_fallback" if getattr(query_cache, "use_local", False) else "disabled"
        )
    )

    # Determine the actual document cache backend at runtime
    doc_cache_backend = "unknown"
    doc_cache_enabled = False
    doc_cache_error = None
    if doc_cache and hasattr(doc_cache, "storage"):
        doc_cache_enabled = True
        # Capture any init error for diagnostics
        doc_cache_error = getattr(doc_cache, "init_error", None)
        backend_class = type(doc_cache.storage).__name__
        if backend_class == "S3StorageBackend":
            doc_cache_backend = (
                "s3" if getattr(doc_cache.storage, "enabled", False) else "s3_disabled"
            )
        elif backend_class == "LocalStorageBackend":
            doc_cache_backend = "local"
        else:
            doc_cache_backend = backend_class
    else:
        # Cache init failed — report what was configured for debugging
        configured_backend = settings.storage_backend
        doc_cache_backend = f"unavailable (configured: {configured_backend})"

    # Check overall liveness / readiness state
    services_status = {
        "postgres_checkpointer": postgres_connected,
        "supabase_company_db": supabase_connected,
        "qdrant_vector_store": qdrant_connected,
        "query_cache_redis": query_cache_status,
        "query_cache_mode": query_cache_mode,
        "document_cache": doc_cache_enabled,
        "document_cache_backend": doc_cache_backend,
        "document_cache_error": doc_cache_error,
    }

    # Only check boolean values — string entries like "disabled" or "local" are always truthy
    boolean_statuses = [v for v in services_status.values() if isinstance(v, bool)]
    any_service_available = any(boolean_statuses)
    health_status = (
        "healthy"
        if (postgres_connected and supabase_connected and qdrant_connected)
        else "degraded"
        if any_service_available
        else "unhealthy"
    )

    return {
        "status": health_status,
        "service": "IDOP — Intelligent Data Operations Platform API",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": settings.app_version,
        "git_commit_sha": settings.git_commit_sha,
        "services": services_status,
        "features_available": {
            "text_to_sql": True,
            "excel_mutations": True,
            "advanced_rag": qdrant_connected,
            "query_routing": True,
        },
        "configuration": {
            "openai_configured": bool(settings.openai_api_key),
            "voyage_configured": bool(settings.voyage_api_key),
            "nomic_configured": bool(settings.nomic_api_key),
            "tavily_configured": bool(settings.tavily_api_key),
            "database_configured": bool(settings.database_url),
            "supabase_configured": bool(settings.supabase_db_url),
            "redis_cache_configured": bool(
                settings.upstash_redis_url and settings.upstash_redis_token
            ),
            "s3_cache_configured": settings.storage_backend == "s3",
        },
        "qdrant_info": collection_info,
        "redis_cache": _format_redis_cache(query_cache),
    }


@router.get(
    "/health/ready", response_model=DetailedReadinessResponse, summary="Readiness check"
)
@track(name="readiness_check")
async def readiness(request: Request) -> dict[str, Any]:
    """
    Readiness probe validating underlying Qdrant, PostgreSQL, and Supabase connections.
    """
    settings = get_settings()

    # Qdrant Check
    vector_store = request.app.state.vector_store
    qdrant_connected = vector_store.health_check()
    collection_info = vector_store.get_collection_info()

    # Postgres Check (checkpointer) — off the event loop
    postgres_connected = False
    try:

        def _check_postgres():
            conn = psycopg2.connect(settings.database_url)
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
            conn.close()
            return True

        postgres_connected = await asyncio.to_thread(_check_postgres)
    except Exception:
        pass

    # Supabase Check (company data) — off the event loop
    supabase_connected = False
    if settings.supabase_db_url:
        try:

            def _check_supabase():
                conn = psycopg2.connect(settings.supabase_db_url)
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                conn.close()
                return True

            supabase_connected = await asyncio.to_thread(_check_supabase)
        except Exception:
            pass

    status = (
        "ready"
        if (qdrant_connected and postgres_connected and supabase_connected)
        else "not_ready"
    )

    return {
        "status": status,
        "qdrant_connected": qdrant_connected,
        "postgres_connected": postgres_connected,
        "supabase_connected": supabase_connected,
        "collection_info": collection_info,
    }


@router.get(
    "/info",
    response_model=SystemInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get system layout and documentation info",
)
@track(name="get_system_info")
async def get_info() -> dict[str, Any]:
    """
    Get system layout, design manuals, operational project phases, and detailed platform endpoint mappings.
    """
    settings = get_settings()
    return {
        "application": {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        },
        "phases": {
            "Phase 1: Foundation": "Completed - Configs, Logging, and Dual-Tier Cache",
            "Phase 2: Text-to-SQL": "Completed - Schema preparation, SQLValidator, LLMJudge, and single-use ApprovalGate",
            "Phase 3: Spreadsheet Mutations": "Completed - OpClassifier, pandas parsing, RuleValidator confirmation gates, and transaction executors",
            "Phase 4: Advanced CSRAG": "Completed - QdrantBM25 hybrid search, HyDE hypothetical query expansion, Reranking, CRAG relevance evaluating, and SRAG verifiers",
            "Phase 5: State Machine Graph": "Completed - 5-path router compilation compiled with local Postgres Saver (STM) and Postgres Store (LTM)",
            "Phase 6: Deployment & Monitoring": "Completed - Production-grade Docker Compose, zero-touch CD orchestration, dynamic Nginx proxying, and Auto-SSL Certbot mapping",
        },
        "features": {
            "router_pathways": ["SQL", "MUTATION", "RAG", "CHAT", "HYBRID"],
            "cache_tier_1": "Upstash Redis query-level caching",
            "cache_tier_2": "S3 / Local filesystem document chunk caching with SHA-256 deduplication",
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "endpoints": {
            "docs": "/docs (Interactive Swagger documentation)",
            "redoc": "/redoc (Alternate ReDoc documentation)",
            "health": "GET /health (Detailed liveness checks)",
            "readiness": "GET /health/ready (Underlying database probes)",
            "info": "GET /info (Comprehensive system layout manual)",
            "stats": "GET /stats (Real-time performance and savings diagnostics)",
            "chat": "POST /chat (Unified 5-path routing conversation endpoint)",
            "upload_doc": "POST /documents/upload (Hybrid ingestion, chunking, and dual-vector indexing)",
            "doc_info": "GET /documents/info (Get Qdrant collection size and details)",
            "sql_generate": "POST /sql/generate (NL-to-SQL Golden Schema prompt generation with pending transaction queue)",
            "sql_approve": "POST /sql/approve (Execute or cancel pending SQL read queries)",
            "mutation_upload": "POST /mutation/upload (Sheet mutation file upload, rule audits, and mapped row preview)",
            "mutation_approve": "POST /mutation/approve (Execute bulk spreadsheet changes inside an all-or-nothing rollback transaction)",
            "cache_stats": "GET /cache/stats (Check document and query cache sizes)",
            "cache_clear": "DELETE /cache/clear (Clear specific document chunks or purge Redis values)",
        },
    }


@router.get(
    "/stats",
    response_model=SystemStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get platform statistics and query cache savings",
)
@track(name="get_system_stats")
async def get_stats(request: Request) -> dict[str, Any]:
    """
    Get system statistics, document ingestion sizes, vector count profiles, and query cache savings estimates.
    """
    settings = get_settings()
    vector_store = request.app.state.vector_store
    collection_info = vector_store.get_collection_info()
    vector_count = collection_info.get("points_count", 0)

    # Fetch document cache stats
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()
    doc_stats = (
        doc_cache.get_cache_stats()
        if doc_cache
        else {
            "total_documents": 0,
            "total_size_human": "0 Bytes",
            "total_size_bytes": 0,
        }
    )

    # Get query cache Redis metrics
    cache_stats = {
        "enabled": False,
        "mode": "disabled",
        "total_estimated_savings": "$0.0000",
        "overall_hit_rate": "0.0%",
    }
    if query_cache and (query_cache.enabled or query_cache.use_local):
        try:
            cache_mode = "redis" if query_cache.enabled else "local_fallback"
            stats = query_cache.get_stats()
            total_cost_saved = 0.0

            # Dynamic retail cost estimations
            cost_estimates = {
                "rag": 0.05,  # $0.05 per GPT-4/RAG query
                "embeddings": 0.0001,  # $0.0001 per embedding check
                "sql_gen": 0.08,  # $0.08 per Golden Text-to-SQL query
                "sql_result": 0.01,  # $0.01 database transaction cost
            }

            for cache_type, cache_data in stats.get("cache_types", {}).items():
                if cache_type in cost_estimates:
                    savings = cache_data.get("hits", 0) * cost_estimates[cache_type]
                    cache_data["estimated_cost_saved"] = f"${savings:.4f}"
                    total_cost_saved += savings

            total_queries = sum(
                c.get("total_queries", 0) for c in stats.get("cache_types", {}).values()
            )
            total_hits = sum(
                c.get("hits", 0) for c in stats.get("cache_types", {}).values()
            )
            hit_rate = (total_hits / max(total_queries, 1)) * 100

            cache_stats = {
                "enabled": True,
                "mode": cache_mode,
                "by_type": stats.get("cache_types", {}),
                "total_estimated_savings": f"${total_cost_saved:.4f}",
                "overall_hit_rate": f"{hit_rate:.1f}%",
            }
        except Exception as e:
            cache_stats = {
                "enabled": True,
                "error": f"Failed to retrieve stats: {e!s}",
            }

    return {
        "indexing": {
            "total_vectors_in_qdrant": vector_count,
            "cached_documents_count": doc_stats.get("total_documents", 0),
            "cached_documents_size": doc_stats.get("total_size_human", "0 Bytes"),
            "cached_documents_size_bytes": doc_stats.get("total_size_bytes", 0),
        },
        "query_cache": cache_stats,
        "system": {
            "checked_at": datetime.now(UTC).isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "configuration": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "cache_ttl": {
                "embeddings": f"{settings.cache_ttl_embeddings}s",
                "rag": f"{settings.cache_ttl_rag}s",
                "sql_generation": f"{settings.cache_ttl_sql_gen}s",
                "sql_results": f"{settings.cache_ttl_sql_result}s",
            },
        },
    }
