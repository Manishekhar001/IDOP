"""
Opik Observability Integration Tests.

Verifies that:
1. The Opik package can be imported and track decorator is available
2. The no-op fallback works when OPIK_API_KEY is not set
3. Environment variable propagation from settings to os.environ works
4. All route modules can import track without errors
5. @track decorators are present on all expected API handlers
6. OPIK_AVAILABLE flag is correctly set
"""

import importlib
import os
import re
import sys

import pytest
from unittest.mock import patch

# ─── Test 1: Opik import and track decorator ─────────────────────────────


def test_opik_package_importable():
    """Verify the opik package is installed and track can be imported."""
    from opik import track

    assert track is not None
    assert callable(track)


def test_app_opik_module_imports():
    """Verify app.opik module loads cleanly.

    When OPIK_TRACK_DISABLE is set to true (e.g. in test env), the real
    Opik SDK import is skipped entirely — OPIK_AVAILABLE is False but
    the no-op track decorator is still callable.
    When OPIK_TRACK_DISABLE is not set, OPIK_AVAILABLE should be True.
    """
    from app.opik import OPIK_AVAILABLE, track as app_track

    opik_disabled = os.environ.get("OPIK_TRACK_DISABLE", "").lower() in ("true", "1")
    if opik_disabled:
        assert (
            OPIK_AVAILABLE is False
        ), "OPIK_AVAILABLE should be False when OPIK_TRACK_DISABLE=true"
    else:
        assert (
            OPIK_AVAILABLE is True
        ), "OPIK_AVAILABLE should be True when opik package is installed and not disabled"
    assert callable(app_track)


# ─── Test 2: No-op fallback behavior ─────────────────────────────────


def test_app_opik_noop_fallback():
    """Verify the no-op fallback works when opik package is unavailable.

    Uses a clean module reload approach: saves the real import state,
    patches to make ``opik`` unimportable, reloads ``app.opik``, then
    restores everything.
    """
    import app.opik as opik_module

    original_track = opik_module.track
    original_available = opik_module.OPIK_AVAILABLE
    # Snapshot env vars so we can restore later
    saved_env = {
        k: os.environ.get(k)
        for k in ["OPIK_API_KEY", "OPIK_PROJECT_NAME", "OPIK_WORKSPACE"]
    }

    try:
        # Simulate opik being unavailable by patching sys.modules so
        # ``from opik import track`` raises ``ImportError``.
        with patch.dict("sys.modules", {"opik": None}):
            if "app.opik" in sys.modules:
                del sys.modules["app.opik"]

            # Use import_module (not reload) because the module was
            # already removed from sys.modules.
            # Reassign opik_module to the freshly imported module so
            # subsequent assertions check the new module (which had the
            # ImportError), not the original one.
            opik_module = importlib.import_module("app.opik")

        assert opik_module.OPIK_AVAILABLE is False

        # The no-op decorator should return the function unchanged
        def sample_func():
            return 42

        decorated = opik_module.track(name="test")(sample_func)
        assert decorated is sample_func, "No-op track should return the same function"
        assert decorated() == 42
    finally:
        # Restore original state on the module that was current when
        # the test started (original_module).  opik_module was re-bound
        # above to the freshly imported module, so save it first.
        import app.opik as original_module  # noqa: F811

        original_module.track = original_track
        original_module.OPIK_AVAILABLE = original_available
        # Restore env vars
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Re-load modules in sys.modules so other tests aren't affected
        if "app.opik" in sys.modules:
            del sys.modules["app.opik"]
        import app.opik  # noqa: F401,F811 — re-import cleanly (side-effect: restores sys.modules)


# ─── Test 3: Env var propagation ────────────────────────────────────


@pytest.mark.usefixtures("monkeypatch")
class TestEnvVarPropagation:
    """Group env-var tests to isolate side effects via monkeypatch + reload."""

    @staticmethod
    def _clean_reload_app_opik():
        """Reload ``app.opik`` after wiping its sys.modules entry."""
        for mod in list(sys.modules.keys()):
            if mod.startswith("app.opik"):
                del sys.modules[mod]
        import app.opik  # noqa: F401

        importlib.reload(app.opik)

    def test_propagates_env_vars_when_configured(self, monkeypatch):
        """Verify OPIK env vars are propagated from settings to os.environ on import.

        Patches ``app.config.get_settings`` (the source of truth for pydantic-settings)
        *before* re-importing ``app.opik`` so the module-level ``settings = get_settings()``
        call picks up the mock.
        """
        # Clear env first
        for key in ["OPIK_API_KEY", "OPIK_PROJECT_NAME", "OPIK_WORKSPACE"]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setattr(
            "app.config.get_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "opik_api_key": "test-api-key-12345",
                    "opik_workspace": "test-workspace",
                    "opik_project_name": "test-project",
                },
            )(),
        )

        self._clean_reload_app_opik()

        assert os.environ.get("OPIK_API_KEY") == "test-api-key-12345"
        assert os.environ.get("OPIK_PROJECT_NAME") == "test-project"
        assert os.environ.get("OPIK_WORKSPACE") == "test-workspace"

    def test_does_not_set_env_vars_when_not_configured(self, monkeypatch):
        """Verify env vars are NOT set when Opik settings are None."""
        for key in ["OPIK_API_KEY", "OPIK_PROJECT_NAME", "OPIK_WORKSPACE"]:
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setattr(
            "app.config.get_settings",
            lambda: type(
                "Settings",
                (),
                {
                    "opik_api_key": None,
                    "opik_workspace": None,
                    "opik_project_name": None,
                },
            )(),
        )

        self._clean_reload_app_opik()

        assert os.environ.get("OPIK_API_KEY") is None
        assert os.environ.get("OPIK_PROJECT_NAME") is None
        assert os.environ.get("OPIK_WORKSPACE") is None


# ─── Test 4: @track decorators on all route modules ──────────────────


def _check_module_has_track_import(module_path: str) -> bool:
    """Check that a module file imports track from app.opik."""
    with open(module_path, encoding="utf-8") as f:
        content = f.read()
    return "from app.opik import track" in content


ROUTE_FILES = [
    "app/api/routes/chat.py",
    "app/api/routes/sql.py",
    "app/api/routes/mutation.py",
    "app/api/routes/documents.py",
    "app/api/routes/cache.py",
    "app/api/routes/memory.py",
    "app/api/routes/health.py",
]


def test_all_route_modules_import_track():
    """Verify every API route module imports @track from app.opik."""
    for rfile in ROUTE_FILES:
        assert _check_module_has_track_import(
            rfile
        ), f"{rfile} is missing 'from app.opik import track'"


def test_route_module_syntax_is_valid():
    """Verify all route modules are syntactically valid Python."""
    for rfile in ROUTE_FILES:
        with open(rfile) as f:
            source = f.read()
        compile(source, rfile, "exec")


# ─── Test 5: @track on key service modules ──────────────────────────

SERVICE_FILES_WITH_TRACK = [
    "app/core/csrag_engine.py",
    "app/core/graph/nodes.py",
    "app/core/embeddings.py",
    "app/core/vector_store.py",
    "app/core/document_processor.py",
    "app/core/memory/stm.py",
    "app/core/memory/ltm.py",
    "app/core/crag/evaluator.py",
    "app/core/crag/web_search.py",
    "app/core/srag/verifier.py",
    "app/core/feature1_sql/vanna_service.py",
    "app/core/feature1_sql/llm_judge.py",
    "app/core/feature1_sql/sql_validator.py",
    "app/core/feature1_sql/executor.py",
    "app/core/feature2_mutation/op_classifier.py",
    "app/core/feature2_mutation/column_mapper.py",
    "app/core/feature2_mutation/llm_judge.py",
    "app/core/feature2_mutation/mutation_generator.py",
    "app/core/feature2_mutation/executor.py",
    "app/core/feature2_mutation/file_parser.py",
    "app/core/feature2_mutation/rule_validator.py",
    "app/core/feature3_rag/hyde.py",
    "app/core/feature3_rag/reranking.py",
    "app/core/feature3_rag/context_enrichment.py",
    "app/core/feature3_rag/ragas_evaluator.py",
]

SERVICE_FILES_WITHOUT_TRACK = [
    "app/core/approval_gate.py",
    "app/services/storage_backend.py",
    "app/services/local_storage.py",
    "app/services/s3_storage.py",
    "app/services/cache_init.py",
    "app/services/query_cache_service.py",
    "app/services/cache_service.py",
    "app/services/pending_store.py",
    "app/core/sparse_vector_service.py",
    "app/core/schema_registry.py",
]


def test_service_modules_with_track():
    """Verify core service modules import @track."""
    for sfile in SERVICE_FILES_WITH_TRACK:
        assert _check_module_has_track_import(
            sfile
        ), f"{sfile} is missing 'from app.opik import track'"


def test_service_modules_without_track():
    """Verify storage/cache backends do NOT import @track (they don't need it)."""
    for sfile in SERVICE_FILES_WITHOUT_TRACK:
        assert not _check_module_has_track_import(
            sfile
        ), f"{sfile} should NOT import track — it's a low-level utility"


# ─── Test 6: @track decorators syntactically valid ───────────────────


def _count_track_decorators(filepath: str) -> int:
    """Count @track(name=...) decorators in a file."""
    with open(filepath) as f:
        content = f.read()
    return len(re.findall(r"@track\(name=.*?\)", content))


def test_track_decorators_present_in_routes():
    """Spot-check that @track decorators exist in route files."""
    for rfile in ROUTE_FILES:
        count = _count_track_decorators(rfile)
        assert count >= 1, f"{rfile} should have at least 1 @track decorator"


# ─── Test 7: OPIK_AVAILABLE flag ─────────────────────────────────────


def test_opik_available_flag():
    """Verify OPIK_AVAILABLE is accessible and boolean."""
    from app.opik import OPIK_AVAILABLE

    assert isinstance(OPIK_AVAILABLE, bool)
