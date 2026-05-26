from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from app.services.cache_service import CacheService
from app.services.query_cache_service import QueryCacheService

router = APIRouter(prefix="/cache", tags=["Cache Management"])

# Services
doc_cache = CacheService()
query_cache = QueryCacheService()


@router.get("/stats", summary="Get stats for both Document and Query caches")
async def get_cache_stats() -> Dict[str, Any]:
    try:
        return {
            "document_cache": doc_cache.get_cache_stats(),
            "query_cache": query_cache.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch cache stats: {str(e)}")


@router.delete("/clear", summary="Clear cache by document ID, or clear all")
async def clear_cache(
    doc_id: Optional[str] = None,
    file_extension: Optional[str] = None,
    clear_query_cache: bool = True
) -> Dict[str, Any]:
    result = {}
    try:
        if doc_id:
            # Clear specific doc chunk cache
            res = doc_cache.clear_cache(doc_id, file_extension)
            result["document_cache"] = res
        else:
            # Clear all doc caches
            res = doc_cache.clear_cache()
            result["document_cache"] = res

        # Clear Redis query cache if requested
        if clear_query_cache:
            success = query_cache.flush_all()
            result["query_cache"] = {"cleared": success, "message": "Query cache flushed completely"}

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
