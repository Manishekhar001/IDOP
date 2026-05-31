"""
Shared TextToSQLService singleton.

Both `app.api.routes.sql` and `app.core.graph.nodes` import from this module
to share the same service instance, ensuring schema training state, query
cache, and pending queries are consistent across the direct API path and
the LangGraph chat path.

This keeps the dependency direction pointing inward (api → core, core → core)
rather than core importing from api (which creates a circular import risk).
"""

from app.core.feature1_sql.vanna_service import TextToSQLService
from app.services.cache_init import get_query_cache
from app.utils.logger import get_logger

logger = get_logger(__name__)

_query_cache = get_query_cache()
sql_service = TextToSQLService(query_cache_service=_query_cache)

logger.info("Shared TextToSQLService initialized")
