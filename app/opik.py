"""
OPIK Observability integration — shared across all route modules.

Inject OPIK environment variables BEFORE importing the SDK so they are
picked up during module initialisation. If OPIK is not installed, no-op
implementations are provided so that @track annotations, start_as_current_trace,
start_as_current_span, and opik_context are always safe to use.
"""

import os
from contextlib import contextmanager

from app.config import get_settings

settings = get_settings()

# Inject OPIK settings into os.environ so the SDK picks them up on import
if getattr(settings, "opik_api_key", None):
    os.environ["OPIK_API_KEY"] = settings.opik_api_key
    if getattr(settings, "opik_workspace", None):
        os.environ["OPIK_WORKSPACE"] = settings.opik_workspace
    if getattr(settings, "opik_project_name", None):
        os.environ["OPIK_PROJECT_NAME"] = settings.opik_project_name

# OPIK monitoring (optional — gracefully handles if not configured)
try:
    from opik import track  # noqa: F401
    from opik import start_as_current_trace, start_as_current_span  # noqa: F401
    from opik import opik_context  # noqa: F401

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False

    # ── No-op decorator ──────────────────────────────────────────────
    def track(name=None, **kwargs):  # type: ignore  # noqa: F811
        def decorator(func):
            return func

        return decorator

    # ── No-op context managers for astream ────────────────────────────
    @contextmanager
    def start_as_current_trace(**kwargs):  # type: ignore  # noqa: F811
        yield

    @contextmanager
    def start_as_current_span(**kwargs):  # type: ignore  # noqa: F811
        yield

    # ── No-op opik_context ───────────────────────────────────────────
    class _FakeOpikContext:
        def update_current_span(self, **kwargs):
            pass

        def update_current_trace(self, **kwargs):
            pass

    opik_context = _FakeOpikContext()  # type: ignore  # noqa: F811
