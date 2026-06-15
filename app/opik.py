"""
OPIK Observability integration — shared across all route modules.

Inject OPIK environment variables BEFORE importing the SDK so they are
picked up during module initialisation. If OPIK is not installed, no-op
implementations are provided so that @track annotations, start_as_current_trace,
start_as_current_span, and opik_context are always safe to use.

Import optimisation:
  Setting OPIK_TRACK_DISABLE=true completely skips importing the real
  opik SDK (which takes ~5s due to OpenTelemetry / httpx / grpcio).
  This is safe because every module that uses @track has already been
  written to handle a no-op decorator.
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

# ──────────────────────────────────────────────────────────────────
# Optimisation: check OPIK_TRACK_DISABLE BEFORE importing the real
# opik SDK.  `opik.config` pulls in OpenTelemetry + httpx + grpcio
# and adds ~5 s to every cold import chain.
# ──────────────────────────────────────────────────────────────────
_OPIK_DISABLED = os.environ.get("OPIK_TRACK_DISABLE", "").lower() in ("true", "1")

if not _OPIK_DISABLED:
    try:
        # OPIK monitoring (optional — gracefully handles if not configured)
        from opik import (
            opik_context,
            start_as_current_span,
            start_as_current_trace,
            track,
        )

        OPIK_AVAILABLE = True
    except ImportError:
        OPIK_AVAILABLE = False
else:
    OPIK_AVAILABLE = False

if not OPIK_AVAILABLE:
    # ── No-op decorator ──────────────────────────────────────────────
    def track(name=None, **kwargs):  # type: ignore
        def decorator(func):
            return func

        return decorator

    # ── No-op context managers for astream ────────────────────────────
    @contextmanager
    def start_as_current_trace(**kwargs):  # type: ignore
        yield

    @contextmanager
    def start_as_current_span(**kwargs):  # type: ignore
        yield

    # ── No-op opik_context ───────────────────────────────────────────
    class _FakeOpikContext:
        def update_current_span(self, **kwargs):
            pass

        def update_current_trace(self, **kwargs):
            pass

    opik_context = _FakeOpikContext()  # type: ignore
