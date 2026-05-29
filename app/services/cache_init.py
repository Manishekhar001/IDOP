"""
Lazy cache initialization module.

Cache services (CacheService, QueryCacheService) are initialized on first use
rather than at module import time. This prevents app startup crashes when
S3/Redis are unreachable — the app starts, and only cache-dependent operations
are affected.
"""

import logging
from typing import Optional
from app.services.cache_service import CacheService
from app.services.query_cache_service import QueryCacheService

logger = logging.getLogger(__name__)

_doc_cache: Optional[CacheService] = None
_query_cache: Optional[QueryCacheService] = None


def get_doc_cache() -> Optional[CacheService]:
    """Lazy-init document cache. Returns None if initialization fails."""
    global _doc_cache
    if _doc_cache is None:
        try:
            _doc_cache = CacheService()
            logger.info("Document cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize document cache: {e}")
            _doc_cache = None
    return _doc_cache


def get_query_cache() -> Optional[QueryCacheService]:
    """Lazy-init query cache. Returns None if initialization fails."""
    global _query_cache
    if _query_cache is None:
        try:
            _query_cache = QueryCacheService()
            logger.info("Query cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize query cache: {e}")
            _query_cache = None
    return _query_cache
