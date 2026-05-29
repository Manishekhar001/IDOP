from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from app.services.cache_init import get_doc_cache, get_query_cache

router = APIRouter(prefix="/cache", tags=["Cache Management"])


@router.get("/stats", summary="Get stats for both Document and Query caches")
async def get_cache_stats() -> Dict[str, Any]:
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()
    try:
        return {
            "document_cache": doc_cache.get_cache_stats() if doc_cache else {"error": "Document cache not initialized", "total_documents": 0},
            "query_cache": query_cache.get_stats() if query_cache else {"enabled": False, "error": "Query cache not initialized"},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch cache stats: {str(e)}")


@router.delete("/clear", summary="Clear cache by document ID, or clear all")
async def clear_cache(
    doc_id: Optional[str] = None,
    file_extension: Optional[str] = None,
    clear_query_cache: bool = True
) -> Dict[str, Any]:
    doc_cache = get_doc_cache()
    query_cache = get_query_cache()
    result = {}
    try:
        if doc_id:
            # Clear specific doc chunk cache
            res = doc_cache.clear_cache(doc_id, file_extension) if doc_cache else {"cleared": False, "message": "Document cache not initialized"}
            result["document_cache"] = res
        else:
            # Clear all doc caches
            res = doc_cache.clear_cache() if doc_cache else {"cleared": False, "message": "Document cache not initialized"}
            result["document_cache"] = res

        # Clear Redis query cache if requested
        if clear_query_cache:
            success = query_cache.flush_all() if query_cache else False
            result["query_cache"] = {"cleared": success, "message": "Query cache flushed completely" if success else "Query cache not initialized or flush failed"}

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
