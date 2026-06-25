"""
Shared pytest fixtures and configuration for the IDOP test suite.

Provides mocked settings, temporary directories, and sample data
fixtures used across all test modules.

IMPORTANT — Settings Strategy:
    Instead of patching app.config.get_settings (which fails because most
    modules bind a local reference via `from app.config import get_settings`),
    we set environment variables BEFORE any production code imports happen.
    The real Settings() class reads from os.environ, so get_settings() returns
    the correct test values regardless of when it is first called.

    Additionally, the get_settings LRU cache is cleared after setting env vars
    to ensure a fresh Settings() object is constructed on first access.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the IDOP project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Environment variable setup — runs at import time, before any test module
# ---------------------------------------------------------------------------
# Assign (not setdefault) to override any real .env values and guarantee
# deterministic settings for all tests.

_TEST_ENV = {
    # General App Config
    "APP_NAME": "IDOP Test Suite",
    "APP_VERSION": "0.1.0-test",
    "ENVIRONMENT": "test",
    "LOG_LEVEL": "WARNING",
    "API_HOST": "127.0.0.1",
    "API_PORT": "8000",
    "ALLOWED_ORIGINS": "*",
    # LLM Provider
    "OPENAI_API_KEY": "sk-test-fake-key-for-unit-tests",
    "LLM_PROVIDER": "groq",
    "LLM_MODEL": "llama-3.1-8b-instant",
    "LLM_TEMPERATURE": "0.0",
    "MEMORY_LLM_MODEL": "llama-3.1-8b-instant",
    "MEMORY_LLM_TEMPERATURE": "0.0",
    "GROQ_API_KEY_1": "gsk-test-fake-groq-key-for-unit-tests",
    # Qdrant
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "test-qdrant-key",
    "COLLECTION_NAME": "idop_test_documents",
    "EMBEDDING_PROVIDER": "nomic",
    # Database
    "DATABASE_URL": "postgresql://test:test@localhost:5432/idop_test",
    "SUPABASE_DB_URL": "",
    # Storage
    "STORAGE_BACKEND": "local",
    "S3_CACHE_BUCKET": "idop-test-bucket",
    "CACHE_DIR": "data/cached_chunks",
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    # Cache TTLs
    "CACHE_TTL_EMBEDDINGS": "604800",
    "CACHE_TTL_RAG": "3600",
    "CACHE_TTL_SQL_GEN": "86400",
    "CACHE_TTL_SQL_RESULT": "900",
    # Search
    "TAVILY_API_KEY": "tvly-test-fake-key",
    "TAVILY_MAX_RESULTS": "5",
    "VOYAGE_API_KEY": "va-test-fake-key",
    # LangGraph
    "STM_MESSAGE_THRESHOLD": "6",
    "CRAG_UPPER_THRESHOLD": "0.7",
    "CRAG_LOWER_THRESHOLD": "0.3",
    "SRAG_MAX_RETRIES": "2",
    "MAX_REWRITE_TRIES": "2",
    "RETRIEVAL_K": "5",
    # Opik — disable tracking entirely to prevent real HTTP calls in tests
    "OPIK_TRACK_DISABLE": "true",
    # Chunking
    "CHUNK_SIZE": "512",
    "CHUNK_OVERLAP": "50",
}

for _key, _val in _TEST_ENV.items():
    os.environ[_key] = _val

# Clear the get_settings LRU cache so it picks up the test env vars on first call
from app.config import (  # noqa: E402 — env vars must be set first
    get_settings as _get_settings,
)

_get_settings.cache_clear()


@pytest.fixture(autouse=True)
def mock_settings():
    """
    Auto-applied fixture that:
    1. Clears the get_settings LRU cache so every test gets a fresh Settings object
    2. Patches database connections for SQL and Mutation approval gates to return None,
       preventing TCP handshake socket timeout delays.
    3. Resets cache singletons to prevent cross-test contamination.

    Returns the active Settings instance (built from the env vars set above).
    """
    from unittest.mock import patch

    from app.services.cache_init import reset_caches
    from app.services.pending_store import reset_pending_store

    reset_caches()  # also clears QueryCacheService._local_cache_shared
    _get_settings.cache_clear()

    # IMPORTANT: reset_pending_store() calls PendingStore.clear() which tries to
    # connect to the database. The patch below ensures _get_connection returns
    # None (no DB available in tests), avoiding a ~30s TCP timeout per call.
    with (
        patch("app.core.approval_gate.ApprovalGate._get_connection", return_value=None),
        patch("app.api.auth._get_connection", return_value=None),
        patch(
            "app.services.pending_store.PendingStore._get_connection", return_value=None
        ),
    ):
        reset_pending_store()
        from app.core.feature1_sql.shared import reset_sql_service

        reset_sql_service()
        yield _get_settings()
    reset_caches()  # also clears QueryCacheService._local_cache_shared
    # reset_pending_store() is called after patching ends but PendingStore now
    # has connect_timeout=2, so even if it tries to connect it fails fast.


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks():
    """Sample document chunks for storage backend tests."""
    return [
        {
            "text": "IDOP is an enterprise-grade intelligent data operations platform.",
            "metadata": {"page": 1, "tokens": 10, "chunk_index": 0},
        },
        {
            "text": "It combines NL-to-SQL, document mutations, and advanced RAG.",
            "metadata": {"page": 1, "tokens": 12, "chunk_index": 1},
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings array (2 chunks x 1536 dimensions)."""
    return np.random.rand(2, 1536).astype(np.float32)


@pytest.fixture
def sample_metadata():
    """Sample document metadata."""
    return {
        "filename": "test_policy.pdf",
        "cached_at": "2026-01-01T00:00:00Z",
        "total_chunks": 2,
        "total_tokens": 22,
    }


@pytest.fixture
def temp_document(tmp_path):
    """Create a temporary test document file."""
    doc_path = tmp_path / "test_document.pdf"
    doc_path.write_text("This is a test document content for IDOP unit testing.")
    return doc_path
