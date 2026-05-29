"""
Unit tests for caching services (CacheService and QueryCacheService).

Tests the document-level cache service with both local and S3 backends,
and the Redis-backed query cache with local in-memory fallback.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.cache_service import CacheService
from app.services.local_storage import LocalStorageBackend
from app.services.query_cache_service import QueryCacheService


# ═══════════════════════════════════════════════════════════════════════
# Tests for CacheService (Document-Level Cache)
# ═══════════════════════════════════════════════════════════════════════

class TestCacheService:
    """Tests for the document cache service wrapping storage backends."""

    @pytest.fixture
    def cache_service(self, tmp_path):
        """Create a CacheService with a local storage backend."""
        backend = LocalStorageBackend(cache_dir=tmp_path)
        return CacheService(storage_backend=backend)

    def test_initialization_with_custom_backend(self, cache_service):
        """Test CacheService accepts a custom storage backend."""
        assert isinstance(cache_service.storage, LocalStorageBackend)

    def test_compute_document_id_sha256(self, cache_service, temp_document):
        """Test that document ID is computed as SHA-256 hash of file content."""
        doc_id = cache_service.compute_document_id(temp_document)

        assert isinstance(doc_id, str)
        assert len(doc_id) == 64  # SHA-256 hex digest length

    def test_compute_document_id_deterministic(self, cache_service, temp_document):
        """Test that the same file always produces the same document ID."""
        id1 = cache_service.compute_document_id(temp_document)
        id2 = cache_service.compute_document_id(temp_document)
        assert id1 == id2

    def test_compute_document_id_missing_file_raises(self, cache_service, tmp_path):
        """Test that a missing file raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError):
            cache_service.compute_document_id(fake_path)

    def test_cache_exists_returns_false_initially(self, cache_service):
        """Test that cache_exists returns False for an uncached document."""
        assert not cache_service.cache_exists("doc_not_cached", "pdf")

    def test_save_and_load_chunks_and_embeddings(self, cache_service, sample_chunks):
        """Test round-trip save and load of chunks + embeddings + metadata."""
        doc_id = "cache_roundtrip_test"
        file_ext = "pdf"
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadata = {"filename": "test.pdf", "total_chunks": 2, "total_tokens": 22}

        cache_service.save_chunks_and_embeddings(
            doc_id, file_ext, sample_chunks, embeddings, metadata
        )

        assert cache_service.cache_exists(doc_id, file_ext)

        result = cache_service.load_chunks_and_embeddings(doc_id, file_ext)
        assert result is not None
        assert result["chunks"] == sample_chunks
        assert result["metadata"] == metadata
        assert len(result["embeddings"]) == 2

    def test_save_mismatched_chunks_and_embeddings_raises(self, cache_service, sample_chunks):
        """Test that mismatched chunk/embedding counts raise ValueError."""
        doc_id = "cache_mismatch"
        file_ext = "pdf"
        embeddings = [[0.1] * 1536]  # Only 1 embedding for 2 chunks

        with pytest.raises(ValueError, match="Chunk/embedding mismatch"):
            cache_service.save_chunks_and_embeddings(
                doc_id, file_ext, sample_chunks, embeddings, {}
            )

    def test_load_uncached_returns_none(self, cache_service):
        """Test that loading an uncached document returns None."""
        result = cache_service.load_chunks_and_embeddings("not_cached_doc", "pdf")
        assert result is None

    def test_get_cache_stats(self, cache_service, sample_chunks):
        """Test getting cache statistics."""
        doc_id = "stats_doc"
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadata = {"filename": "test.pdf", "total_chunks": 2}

        cache_service.save_chunks_and_embeddings(
            doc_id, "pdf", sample_chunks, embeddings, metadata
        )

        stats = cache_service.get_cache_stats()
        assert stats["backend"] == "local"
        assert stats["total_documents"] >= 1

    def test_clear_specific_document_cache(self, cache_service, sample_chunks):
        """Test clearing cache for a specific document."""
        doc_id = "clear_specific"
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadata = {"filename": "test.pdf", "total_chunks": 2}

        cache_service.save_chunks_and_embeddings(
            doc_id, "pdf", sample_chunks, embeddings, metadata
        )
        assert cache_service.cache_exists(doc_id, "pdf")

        result = cache_service.clear_cache(doc_id=doc_id, file_extension="pdf")
        assert result["cleared"] is True
        assert result["documents_cleared"] == 1
        assert not cache_service.cache_exists(doc_id, "pdf")

    def test_clear_entire_cache(self, cache_service, sample_chunks):
        """Test clearing the entire document cache."""
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadata = {"filename": "test.pdf", "total_chunks": 2}

        for i in range(3):
            cache_service.save_chunks_and_embeddings(
                f"bulk_doc_{i}", "pdf", sample_chunks, embeddings, metadata
            )

        result = cache_service.clear_cache()
        assert result["cleared"] is True

    def test_clear_requires_extension_for_specific_doc(self, cache_service):
        """Test that clearing a specific doc without file_extension returns error."""
        result = cache_service.clear_cache(doc_id="some_doc")
        assert result["cleared"] is False
        assert "file_extension required" in result["message"]


# ═══════════════════════════════════════════════════════════════════════
# Tests for QueryCacheService (Redis / Local In-Memory Fallback)
# ═══════════════════════════════════════════════════════════════════════

class TestQueryCacheService:
    """Tests for the Redis-backed query cache with local fallback."""

    @pytest.fixture
    def local_cache(self):
        """Create a QueryCacheService in local in-memory fallback mode."""
        # Hermetically mock get_settings inside the target module to prevent
        # reading real credentials from the local .env file.
        with patch("app.services.query_cache_service.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.upstash_redis_url = None
            mock_settings.upstash_redis_token = None
            mock_get.return_value = mock_settings
            return QueryCacheService(redis_url=None, redis_token=None)

    def test_initialization_local_fallback(self, local_cache):
        """Test that the service initializes in local fallback mode without Redis."""
        assert local_cache.use_local is True
        assert local_cache.enabled is False
        assert local_cache._local_cache == {}

    def test_set_and_get_local_cache(self, local_cache):
        """Test storing and retrieving a value from local cache."""
        key = "test:key:123"
        value = {"answer": "IDOP is an enterprise data operations platform."}

        success = local_cache.set(key, value, ttl=3600, cache_type="rag")
        assert success is True

        result = local_cache.get(key, cache_type="rag")
        assert result is not None
        assert result["answer"] == value["answer"]

    def test_get_miss_local_cache(self, local_cache):
        """Test that a cache miss returns None."""
        result = local_cache.get("nonexistent:key", cache_type="rag")
        assert result is None

    def test_embedding_key_generation(self, local_cache):
        """Test embedding cache key uses SHA-256 of the text."""
        key = local_cache.get_embedding_key("What is IDOP?")
        assert key.startswith("embedding:")
        assert len(key) == len("embedding:") + 64

    def test_rag_key_generation(self, local_cache):
        """Test RAG cache key includes question hash and top_k."""
        key = local_cache.get_rag_key("What is our refund policy?", top_k=4)
        assert key.startswith("rag:")
        assert ":4" in key

    def test_sql_gen_key_generation(self, local_cache):
        """Test SQL generation cache key uses question hash."""
        key = local_cache.get_sql_gen_key("Show total revenue by quarter")
        assert key.startswith("sql_gen:")

    def test_sql_result_key_generation(self, local_cache):
        """Test SQL result cache key normalizes whitespace and case."""
        key1 = local_cache.get_sql_result_key("SELECT * FROM products")
        key2 = local_cache.get_sql_result_key("  select  *  from  products  ")
        assert key1 == key2  # Normalization should produce identical keys

    def test_cache_hit_tracking(self, local_cache):
        """Test that cache hits and misses are tracked in statistics."""
        local_cache.set("tracked:key", {"data": "value"}, ttl=3600, cache_type="rag")

        local_cache.get("tracked:key", cache_type="rag")
        local_cache.get("missing:key", cache_type="rag")

        stats = local_cache.get_stats()
        rag_stats = stats["cache_types"]["rag"]
        assert rag_stats["hits"] == 1
        assert rag_stats["misses"] == 1
        assert rag_stats["total_queries"] == 2
        assert rag_stats["hit_rate"] == "50.0%"

    def test_reset_stats(self, local_cache):
        """Test resetting cache statistics."""
        local_cache.set("res:key", {"data": "val"}, ttl=3600, cache_type="rag")
        local_cache.get("res:key", cache_type="rag")

        local_cache.reset_stats()

        stats = local_cache.get_stats()
        assert stats["cache_types"]["rag"]["hits"] == 0
        assert stats["cache_types"]["rag"]["misses"] == 0

    def test_health_check_local_mode(self, local_cache):
        """Test health check reports healthy in local fallback mode."""
        health = local_cache.health_check()
        assert health["status"] == "healthy"
        assert health["mode"] == "local"

    def test_multiple_cache_types_tracked_independently(self, local_cache):
        """Test that different cache types (rag, sql_gen, etc.) track stats independently."""
        local_cache.set("rag:k1", {"a": 1}, ttl=3600, cache_type="rag")
        local_cache.set("sql:k1", {"b": 2}, ttl=86400, cache_type="sql_gen")

        local_cache.get("rag:k1", cache_type="rag")
        local_cache.get("sql:k1", cache_type="sql_gen")
        local_cache.get("sql:miss", cache_type="sql_gen")

        stats = local_cache.get_stats()
        assert stats["cache_types"]["rag"]["hits"] == 1
        assert stats["cache_types"]["rag"]["misses"] == 0
        assert stats["cache_types"]["sql_gen"]["hits"] == 1
        assert stats["cache_types"]["sql_gen"]["misses"] == 1

    def test_flush_all_local_cache(self, local_cache):
        """Test that flush_all clears all keys from local cache."""
        # Flush first to start clean (shared class-level cache may have data from other tests)
        local_cache.flush_all()
        local_cache.set("flush:k1", {"a": 1}, ttl=3600)
        local_cache.set("flush:k2", {"b": 2}, ttl=3600)
        assert "flush:k1" in local_cache._local_cache
        assert "flush:k2" in local_cache._local_cache
        assert local_cache.flush_all() is True
        assert len(local_cache._local_cache) == 0

    def test_delete_disabled(self, local_cache):
        """Test that delete returns 0 when Redis is disabled."""
        assert local_cache.delete("some:pattern:*") == 0

    def test_serialization_complex_data(self, local_cache):
        """Test that complex nested data structures survive serialization round-trip."""
        complex_value = {
            "answer": "The refund policy states...",
            "sources": [{"file": "policy.pdf", "page": 3}],
            "confidence": 0.95,
            "metadata": {"model": "gpt-4o", "tokens": 150},
        }
        local_cache.set("complex:key", complex_value, ttl=3600)
        result = local_cache.get("complex:key")
        assert result == complex_value
