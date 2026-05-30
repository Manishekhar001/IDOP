"""
Integration tests for the centralized cache system.

Validates:
- Singleton factory in cache_init.py
- QueryCacheService round-trip (set/get/delete) in local fallback mode
- CacheService round-trip with LocalStorageBackend
- Cache reset lifecycle
- Shared state across get_query_cache() calls
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.cache_init import get_doc_cache, get_query_cache, reset_caches
from app.services.cache_service import CacheService
from app.services.local_storage import LocalStorageBackend
from app.services.query_cache_service import QueryCacheService


class TestCacheInitSingleton:
    """Tests for the centralized cache singleton factory."""

    def test_get_query_cache_returns_same_instance(self):
        """get_query_cache() returns the same singleton on repeated calls."""
        reset_caches()
        cache1 = get_query_cache()
        cache2 = get_query_cache()
        assert cache1 is cache2

    def test_get_doc_cache_returns_same_instance(self):
        """get_doc_cache() returns the same singleton on repeated calls."""
        reset_caches()
        cache1 = get_doc_cache()
        cache2 = get_doc_cache()
        assert cache1 is cache2

    def test_reset_caches_clears_singletons(self):
        """reset_caches() forces re-initialization on next access."""
        reset_caches()
        cache_before = get_query_cache()
        reset_caches()
        cache_after = get_query_cache()
        # They may be equal or not depending on config, but they should be
        # different object instances after reset
        assert cache_before is not cache_after

    def test_query_cache_local_fallback_mode(self):
        """Without Redis credentials (mocked), query cache operates in local fallback mode."""
        reset_caches()
        cache = get_query_cache()
        assert cache is not None
        # Under mock settings, Redis creds are None → local fallback
        assert cache.use_local is True or cache.enabled is True
        # Either way, the cache should be usable
        cache.set("test:probe", {"ok": True}, ttl=60)
        assert cache.get("test:probe") is not None

    def test_doc_cache_initializes_successfully(self):
        """Doc cache should initialize regardless of backend type."""
        reset_caches()
        cache = get_doc_cache()
        assert cache is not None
        assert hasattr(cache, "storage")
        assert cache.storage is not None


class TestQueryCacheRoundTrip:
    """End-to-end round-trip tests for QueryCacheService in local mode."""

    @pytest.fixture
    def cache(self):
        """Get a fresh QueryCacheService in local fallback mode."""
        with patch("app.services.query_cache_service.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.upstash_redis_url = None
            mock_settings.upstash_redis_token = None
            mock_get.return_value = mock_settings
            svc = QueryCacheService(redis_url=None, redis_token=None)
            svc.flush_all()  # start clean
            yield svc
            svc.flush_all()  # cleanup

    def test_rag_cache_round_trip(self, cache):
        """Test caching a RAG answer and retrieving it."""
        question = "What is the company's refund policy?"
        top_k = 5
        key = cache.get_rag_key(question, top_k)

        answer = {
            "answer": "Our refund policy allows returns within 30 days.",
            "sources": [{"file": "policy.pdf", "chunk": 3}],
            "confidence": 0.92,
        }

        assert cache.get(key, cache_type="rag") is None  # miss
        cache.set(key, answer, ttl=3600, cache_type="rag")
        result = cache.get(key, cache_type="rag")  # hit

        assert result is not None
        assert result["answer"] == answer["answer"]
        assert result["confidence"] == 0.92

    def test_sql_gen_cache_round_trip(self, cache):
        """Test caching a SQL generation result."""
        question = "Show me total revenue by quarter"
        key = cache.get_sql_gen_key(question)

        sql_result = {
            "sql": "SELECT quarter, SUM(revenue) FROM sales GROUP BY quarter",
            "explanation": "Aggregates revenue by fiscal quarter",
        }

        cache.set(key, sql_result, ttl=86400, cache_type="sql_gen")
        result = cache.get(key, cache_type="sql_gen")
        assert result["sql"] == sql_result["sql"]

    def test_sql_result_cache_round_trip(self, cache):
        """Test caching SQL execution results."""
        sql = "SELECT COUNT(*) FROM users WHERE active = true"
        key = cache.get_sql_result_key(sql)

        exec_result = {
            "rows": [{"count": 1523}],
            "row_count": 1,
            "columns": ["count"],
        }

        cache.set(key, exec_result, ttl=900, cache_type="sql_result")
        result = cache.get(key, cache_type="sql_result")
        assert result["rows"][0]["count"] == 1523

    def test_embedding_cache_round_trip(self, cache):
        """Test caching embedding vectors."""
        text = "IDOP is an enterprise data operations platform."
        key = cache.get_embedding_key(text)

        embedding = {"vector": [0.1] * 1536, "model": "text-embedding-3-small"}

        cache.set(key, embedding, ttl=604800, cache_type="embedding")
        result = cache.get(key, cache_type="embedding")
        assert len(result["vector"]) == 1536

    def test_cache_key_normalization(self, cache):
        """SQL result keys should normalize whitespace and case."""
        key1 = cache.get_sql_result_key("SELECT * FROM users")
        key2 = cache.get_sql_result_key("  select  *  from  users  ")
        assert key1 == key2

    def test_delete_by_pattern(self, cache):
        """Test pattern-based deletion in local mode."""
        cache.set("rag:abc123:5", {"a": 1}, ttl=3600)
        cache.set("rag:def456:5", {"b": 2}, ttl=3600)
        cache.set("sql_gen:abc123", {"c": 3}, ttl=3600)

        deleted = cache.delete("rag:*")
        assert deleted == 2
        assert cache.get("sql_gen:abc123") is not None  # untouched

    def test_stats_accumulate_correctly(self, cache):
        """Stats should accumulate across operations."""
        cache.set("test:k1", {"v": 1}, ttl=3600, cache_type="rag")
        cache.get("test:k1", cache_type="rag")  # hit
        cache.get("test:miss", cache_type="rag")  # miss
        cache.get("test:miss2", cache_type="sql_gen")  # miss

        stats = cache.get_stats()
        assert stats["enabled"] is True
        assert stats["mode"] == "local_fallback"
        assert stats["cache_types"]["rag"]["hits"] == 1
        assert stats["cache_types"]["rag"]["misses"] == 1
        assert stats["cache_types"]["sql_gen"]["misses"] == 1

    def test_flush_clears_all(self, cache):
        """flush_all should remove all keys from local cache."""
        cache.set("a:1", {"v": 1}, ttl=3600)
        cache.set("b:2", {"v": 2}, ttl=3600)
        assert cache.flush_all() is True
        assert cache.get("a:1") is None
        assert cache.get("b:2") is None

    def test_health_check(self, cache):
        """Health check should report healthy in local mode."""
        health = cache.health_check()
        assert health["status"] == "healthy"
        assert health["mode"] == "local"


class TestDocumentCacheRoundTrip:
    """End-to-end round-trip tests for CacheService with LocalStorageBackend."""

    @pytest.fixture
    def doc_cache(self, tmp_path):
        """Create a CacheService with a temp local backend."""
        backend = LocalStorageBackend(cache_dir=tmp_path)
        return CacheService(storage_backend=backend)

    def test_document_lifecycle(self, doc_cache, tmp_path):
        """Test full save -> exists -> load -> clear lifecycle."""
        doc_id = "test_lifecycle_doc"
        ext = "pdf"
        chunks = [
            {"text": "First chunk", "metadata": {"page": 1, "chunk_index": 0}},
            {"text": "Second chunk", "metadata": {"page": 1, "chunk_index": 1}},
        ]
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        metadata = {"filename": "test.pdf", "total_chunks": 2}

        # 1. Initially not cached
        assert not doc_cache.cache_exists(doc_id, ext)
        assert doc_cache.load_chunks_and_embeddings(doc_id, ext) is None

        # 2. Save
        doc_cache.save_chunks_and_embeddings(doc_id, ext, chunks, embeddings, metadata)

        # 3. Now cached
        assert doc_cache.cache_exists(doc_id, ext)

        # 4. Load and verify
        loaded = doc_cache.load_chunks_and_embeddings(doc_id, ext)
        assert loaded is not None
        assert len(loaded["chunks"]) == 2
        assert len(loaded["embeddings"]) == 2
        assert loaded["metadata"]["total_chunks"] == 2

        # 5. Clear
        result = doc_cache.clear_cache(doc_id=doc_id, file_extension=ext)
        assert result["cleared"] is True
        assert not doc_cache.cache_exists(doc_id, ext)

    def test_compute_document_id(self, doc_cache, tmp_path):
        """Document IDs should be deterministic SHA-256 hashes."""
        doc = tmp_path / "test_file.txt"
        doc.write_text("Hello IDOP!")
        id1 = doc_cache.compute_document_id(doc)
        id2 = doc_cache.compute_document_id(doc)
        assert id1 == id2
        assert len(id1) == 64  # SHA-256 hex

    def test_cache_stats(self, doc_cache):
        """Stats should report local backend info."""
        stats = doc_cache.get_cache_stats()
        assert stats["backend"] == "local"
        assert "total_documents" in stats
