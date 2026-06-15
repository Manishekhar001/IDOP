"""
Lazy cache initialization module — centralised singleton factory.

All modules that need CacheService or QueryCacheService should import from here
rather than instantiating their own copies. This ensures shared statistics,
shared in-memory fallback data, and consistent lifecycle management.

Usage:
    from app.services.cache_init import get_doc_cache, get_query_cache
    cache = get_query_cache()          # returns the singleton or None
"""

import logging

from app.services.cache_service import CacheService
from app.services.query_cache_service import QueryCacheService

logger = logging.getLogger(__name__)

_doc_cache: CacheService | None = None
_query_cache: QueryCacheService | None = None
_doc_cache_init_attempted = False
_query_cache_init_attempted = False


def get_doc_cache() -> CacheService | None:
    """Lazy-init document cache. Returns None if initialization fails."""
    global _doc_cache, _doc_cache_init_attempted
    if _doc_cache is None and not _doc_cache_init_attempted:
        _doc_cache_init_attempted = True
        try:
            _doc_cache = CacheService()
            backend_type = type(_doc_cache.storage).__name__
            logger.info(f"Document cache initialized (backend: {backend_type})")
        except Exception as e:
            logger.error(f"Failed to initialize document cache: {e}", exc_info=True)
            _doc_cache = None
    return _doc_cache


def get_query_cache() -> QueryCacheService | None:
    """Lazy-init query cache. Returns None if initialization fails."""
    global _query_cache, _query_cache_init_attempted
    if _query_cache is None and not _query_cache_init_attempted:
        _query_cache_init_attempted = True
        try:
            _query_cache = QueryCacheService()
            mode = (
                "redis"
                if _query_cache.enabled
                else ("local_fallback" if _query_cache.use_local else "disabled")
            )
            logger.info(f"Query cache initialized (mode: {mode})")
        except Exception as e:
            logger.error(f"Failed to initialize query cache: {e}", exc_info=True)
            _query_cache = None
    return _query_cache


def reset_caches() -> None:
    """Reset both cache singletons and clear in-memory cache data. Useful for testing or hot-reloading config."""
    global _doc_cache, _query_cache, _doc_cache_init_attempted, _query_cache_init_attempted
    _doc_cache = None
    _query_cache = None
    _doc_cache_init_attempted = False
    _query_cache_init_attempted = False
    QueryCacheService.reset_local_cache()
    logger.info("Cache singletons reset — next access will re-initialize")
