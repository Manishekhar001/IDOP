"""
Shared TextToSQLService singleton — lazy-initialized.

Both `app.api.routes.sql` and `app.core.graph.nodes` import from this module
to share the same service instance, ensuring schema training state, query
cache, and pending queries are consistent across the direct API path and
the LangGraph chat path.

This keeps the dependency direction pointing inward (api → core, core → core)
rather than core importing from api (which creates a circular import risk).

Lazy initialisation
-------------------
The VannaAgentWrapper is NOT created at module-import time.  It is created on
the first call to any attribute of `sql_service`.  This avoids a chain of
heavy imports (vanna → vanna.integrations → langchain → ...) during the
uvicorn startup lifespan.  If Vanna or any sub-import stutters, the error is
deferred until the service is actually used, rather than corrupting the
entire startup chain.
"""

from app.utils.logger import get_logger

logger = get_logger(__name__)


class _LazyTextToSQLService:
    """Lazily-initialised proxy for TextToSQLService.

    Stores the singleton on first attribute access so that the heavy Vanna
    import chain (vanna → vanna.integrations → langchain → …) does not run
    at module-import time.  This keeps the startup lifespan fast and robust.
    """

    def __init__(self):
        self._instance = None

    def _ensure(self):
        if self._instance is None:
            from app.core.feature1_sql.vanna_service import TextToSQLService
            from app.services.cache_init import get_query_cache

            _query_cache = get_query_cache()
            self._instance = TextToSQLService(query_cache_service=_query_cache)
            logger.info("Lazy TextToSQLService initialised on first access")
        return self._instance

    def __getattr__(self, name: str):
        # Called only for attributes that don't exist on _LazyTextToSQLService
        return getattr(self._ensure(), name)


# Proxy object — acts like a TextToSQLService singleton but delays the
# actual constructor call (and its heavy import chain) until first use.
sql_service = _LazyTextToSQLService()

logger.info("Lazy TextToSQLService proxy ready (instance not yet created)")
