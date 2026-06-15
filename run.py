#!/usr/bin/env python3
"""
IDOP Application Entry Point
-----------------------------
Sets the Windows event loop policy to SelectorEventLoop BEFORE importing
uvicorn. On Windows, uvicorn creates a ProactorEventLoop at import time,
and psycopg raises::

    InterfaceError: Psycopg cannot use the 'ProactorEventLoop'

The module-level fix in app/main.py cannot work for ``uvicorn app.main:app``
(CLI mode) because uvicorn creates the ProactorEventLoop **before** the
import chain executes.

Use this script instead of ``uvicorn app.main:app`` for local dev on Windows:

    python run.py          # development (hot-reload enabled)
    python run.py --prod   # production (no hot-reload)

On Linux/macOS this script is harmless — the policy is already correct.
"""

import sys

if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    reload_enabled = "--prod" not in sys.argv

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=reload_enabled,
        log_level=settings.log_level.lower(),
    )
