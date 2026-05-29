"""
Shared pending operations store — centralised singleton dicts.

Both the LangGraph graph nodes (nodes.py) and the API routes (sql.py, mutation.py)
need to read/write pending queries and mutations.  Using a shared module here
avoids the instance-isolation bug where the graph node creates its own local
TextToSQLService whose pending_queries dict is invisible to the route.

Usage:
    from app.services.pending_store import pending_queries, pending_mutations
    pending_queries[query_id] = {"sql": "...", "status": "pending_approval", ...}
"""

from typing import Any, Dict

# Shared pending SQL queries — used by graph nodes AND /sql routes
pending_queries: Dict[str, Dict[str, Any]] = {}

# Shared pending mutation sessions — used by mutation route
pending_mutations: Dict[str, Dict[str, Any]] = {}
