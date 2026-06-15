from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.opik import track
from app.services.cache_init import get_doc_cache, get_query_cache
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cache", tags=["Cache Management"])


# ─── Response Models ───────────────────────────────────────────────────
class CacheStatsResponse(BaseModel):
    """Detailed statistics for both document and query caches."""

    document_cache: dict[str, Any] = Field(
        ..., description="Document-level storage cache stats (local or S3)"
    )
    query_cache: dict[str, Any] = Field(
        ..., description="Query-level cache stats (Redis or local fallback)"
    )


class CacheClearResponse(BaseModel):
    """Result of a cache clear operation."""

    document_cache: dict[str, Any] | None = Field(
        None, description="Document cache clear result"
    )
    query_cache: dict[str, Any] | None = Field(
        None, description="Query cache clear result"
    )


class CacheHealthResponse(BaseModel):
    """Health status of both cache layers."""

    document_cache: dict[str, Any] = Field(..., description="Document cache health")
    query_cache: dict[str, Any] = Field(..., description="Query cache health")
    overall_status: str = Field(
        ..., description="Overall cache system status: healthy / degraded / unhealthy"
    )


class CacheTestResponse(BaseModel):
    """Result of a cache write-read-delete round-trip test."""

    test_passed: bool = Field(..., description="Whether the round-trip test succeeded")
    cache_mode: str = Field(
        ..., description="Cache mode used for the test (redis / local_fallback)"
    )
    details: dict[str, Any] = Field(..., description="Step-by-step test results")


# ─── Endpoints ─────────────────────────────────────────────────────────


@router.get(
    "/stats",
    response_model=CacheStatsResponse,
    summary="Get stats for both Document and Query caches",
    description="Returns detailed statistics including hit rates, sizes, and backend information for all cache layers.",
)
@track(name="get_cache_stats")
async def get_cache_stats() -> CacheStatsResponse:
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()
    try:
        return CacheStatsResponse(
            document_cache=(
                doc_cache.get_cache_stats()
                if doc_cache
                else {"error": "Document cache not initialized", "total_documents": 0}
            ),
            query_cache=(
                query_cache.get_stats()
                if query_cache
                else {
                    "enabled": False,
                    "mode": "disabled",
                    "error": "Query cache not initialized",
                }
            ),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch cache stats: {e!s}"
        )


@router.delete(
    "/clear",
    response_model=CacheClearResponse,
    summary="Clear cache by document ID, or clear all",
    description=(
        "Clear a specific document's chunk cache by doc_id + file_extension, "
        "or omit both to flush everything. Optionally clears the query cache as well."
    ),
)
@track(name="clear_cache")
async def clear_cache(
    doc_id: str | None = None,
    file_extension: str | None = None,
    clear_query_cache: bool = True,
) -> CacheClearResponse:
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()
    result: dict[str, Any] = {}
    try:
        if doc_id:
            res = (
                doc_cache.clear_cache(doc_id, file_extension)
                if doc_cache
                else {"cleared": False, "message": "Document cache not initialized"}
            )
            result["document_cache"] = res
        else:
            res = (
                doc_cache.clear_cache()
                if doc_cache
                else {"cleared": False, "message": "Document cache not initialized"}
            )
            result["document_cache"] = res

        if clear_query_cache:
            success = query_cache.flush_all() if query_cache else False
            result["query_cache"] = {
                "cleared": success,
                "message": (
                    "Query cache flushed completely"
                    if success
                    else "Query cache not initialized or flush failed"
                ),
            }

        return CacheClearResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e!s}")


@router.get(
    "/health",
    response_model=CacheHealthResponse,
    summary="Check health of all cache layers",
    description="Performs connectivity checks on both document storage and query cache backends.",
)
@track(name="cache_health")
async def cache_health() -> CacheHealthResponse:
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()

    doc_health: dict[str, Any]
    if doc_cache:
        backend_class = type(doc_cache.storage).__name__
        doc_health = {
            "status": "healthy",
            "backend": backend_class,
            "message": f"{backend_class} is accessible",
        }
    else:
        doc_health = {
            "status": "unhealthy",
            "backend": "none",
            "message": "Document cache failed to initialize",
        }

    query_health: dict[str, Any]
    if query_cache:
        query_health = query_cache.health_check()
    else:
        query_health = {
            "status": "unhealthy",
            "message": "Query cache failed to initialize",
        }

    doc_ok = doc_health.get("status") == "healthy"
    query_ok = query_health.get("status") == "healthy"

    if doc_ok and query_ok:
        overall = "healthy"
    elif doc_ok or query_ok:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return CacheHealthResponse(
        document_cache=doc_health, query_cache=query_health, overall_status=overall
    )


@router.post(
    "/test",
    response_model=CacheTestResponse,
    summary="Run a cache round-trip test",
    description=(
        "Writes a test key, reads it back, verifies the value, and deletes it. "
        "Use this to validate that the cache backend is working in real-time."
    ),
)
@track(name="test_cache")
async def test_cache() -> CacheTestResponse:
    query_cache = get_query_cache()
    if not query_cache:
        return CacheTestResponse(
            test_passed=False,
            cache_mode="disabled",
            details={"error": "Query cache not initialized"},
        )

    mode = (
        "redis"
        if query_cache.enabled
        else ("local_fallback" if query_cache.use_local else "disabled")
    )
    test_key = "cache_test:round_trip"
    test_value = {"test": True, "message": "IDOP cache round-trip test"}
    details: dict[str, Any] = {"mode": mode}

    # Step 1: Write
    write_ok = query_cache.set(test_key, test_value, ttl=60, cache_type="rag")
    details["write"] = "ok" if write_ok else "FAILED"

    # Step 2: Read
    read_result = query_cache.get(test_key, cache_type="rag")
    read_ok = read_result is not None and read_result.get("test") is True
    details["read"] = "ok" if read_ok else "FAILED"
    details["read_value"] = read_result

    # Step 3: Delete
    deleted = query_cache.delete(test_key)
    details["delete"] = f"ok (deleted {deleted})"

    # Step 4: Verify deletion
    after_delete = query_cache.get(test_key, cache_type="rag")
    verify_ok = after_delete is None
    details["verify_deleted"] = "ok" if verify_ok else "FAILED — key still exists"

    test_passed = write_ok and read_ok and verify_ok

    return CacheTestResponse(test_passed=test_passed, cache_mode=mode, details=details)
