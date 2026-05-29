"""
Shared pytest fixtures and configuration for the IDOP test suite.

Provides mocked settings, temporary directories, and sample data
fixtures used across all test modules.
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

# Ensure the IDOP project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set dummy environment variables to prevent Pydantic ValidationError on Settings instantiation
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-qdrant-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/idop_test")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-fake-key")



# ---------------------------------------------------------------------------
# Mock Settings — avoids needing a real .env file during tests
# ---------------------------------------------------------------------------

class MockSettings:
    """Minimal mock of app.config.Settings for offline testing."""
    app_name = "IDOP Test Suite"
    app_version = "0.1.0-test"
    environment = "test"
    log_level = "WARNING"
    api_host = "127.0.0.1"
    api_port = 8000
    allowed_origins = "*"

    openai_api_key = "sk-test-fake-key-for-unit-tests"
    llm_model = "gpt-4o"
    llm_temperature = 0.0
    memory_llm_model = "gpt-4o-mini"
    memory_llm_temperature = 0.0

    qdrant_url = "http://localhost:6333"
    qdrant_api_key = "test-qdrant-key"
    collection_name = "idop_test_documents"
    embedding_dimension = 1536

    database_url = "postgresql://test:test@localhost:5432/idop_test"

    storage_backend = "local"
    s3_cache_bucket = "idop-test-bucket"
    aws_region = "us-east-1"
    aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
    aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

    upstash_redis_url = None
    upstash_redis_token = None

    cache_ttl_embeddings = 604800
    cache_ttl_rag = 3600
    cache_ttl_sql_gen = 86400
    cache_ttl_sql_result = 900

    tavily_api_key = "tvly-test-fake-key"
    tavily_max_results = 5
    voyage_api_key = "va-test-fake-key"

    stm_message_threshold = 6
    crag_upper_threshold = 0.7
    crag_lower_threshold = 0.3
    srag_max_retries = 2
    max_rewrite_tries = 2

    chunk_size = 512
    chunk_overlap = 50


@pytest.fixture(autouse=True)
def mock_settings():
    """
    Auto-applied fixture that patches get_settings() globally,
    so no test ever needs a real .env or external credentials.
    Also patches database connections for SQL and Mutation approval gates
    to return None, preventing TCP handshake socket timeout delays.
    """
    mock = MockSettings()
    with patch("app.config.get_settings", return_value=mock), \
         patch("app.core.feature1_sql.approval_gate.ApprovalGate._get_connection", return_value=None), \
         patch("app.core.feature2_mutation.approval_gate.MutationApprovalGate._get_connection", return_value=None):
        yield mock


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
