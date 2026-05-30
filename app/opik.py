"""
OPIK Observability integration — shared across all route modules.

Inject OPIK environment variables BEFORE importing the SDK so they are
picked up during module initialisation. If OPIK is not installed, a no-op
decorator is provided so that @track annotations are always safe to use.
"""

import os

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

    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False

    # Create a no-op decorator if OPIK is not installed
    def track(name=None, **kwargs):  # type: ignore  # noqa: F811
        def decorator(func):
            return func

        return decorator
