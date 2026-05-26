"""
Unit tests for storage backends (LocalStorage and S3Storage).

Tests both local filesystem and S3 storage implementations to ensure
they correctly implement the StorageBackend interface for the IDOP
document caching pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.local_storage import LocalStorageBackend
from app.services.s3_storage import S3StorageBackend


# ═══════════════════════════════════════════════════════════════════════
# Tests for LocalStorageBackend
# ═══════════════════════════════════════════════════════════════════════

class TestLocalStorageBackend:
    """Tests for the local filesystem storage backend."""

    @pytest.fixture
    def local_storage(self, tmp_path):
        """Create a LocalStorageBackend with a temporary directory."""
        return LocalStorageBackend(cache_dir=tmp_path)

    def test_initialization(self, local_storage, tmp_path):
        """Test that LocalStorageBackend initializes with correct cache directory."""
        assert local_storage.cache_dir == tmp_path
        assert local_storage.cache_dir.exists()

    def test_save_and_load_chunks(self, local_storage, sample_chunks):
        """Test round-trip save and load of chunks.json."""
        doc_id = "test_doc_abc123"
        file_ext = "pdf"

        local_storage.save_chunks(doc_id, file_ext, sample_chunks)

        chunks_file = local_storage._get_document_path(doc_id) / "chunks.json"
        assert chunks_file.exists()

        loaded = local_storage.load_chunks(doc_id, file_ext)
        assert loaded == sample_chunks
        assert len(loaded) == 2
        assert loaded[0]["text"] == sample_chunks[0]["text"]

    def test_save_and_load_embeddings(self, local_storage, sample_embeddings):
        """Test round-trip save and load of embeddings.npy."""
        doc_id = "test_doc_emb456"
        file_ext = "pdf"

        local_storage.save_embeddings(doc_id, file_ext, sample_embeddings)

        embeddings_file = local_storage._get_document_path(doc_id) / "embeddings.npy"
        assert embeddings_file.exists()

        loaded = local_storage.load_embeddings(doc_id, file_ext)
        assert np.array_equal(loaded, sample_embeddings)
        assert loaded.shape == (2, 1536)

    def test_save_and_load_metadata(self, local_storage, sample_metadata):
        """Test round-trip save and load of metadata.json."""
        doc_id = "test_doc_meta789"
        file_ext = "pdf"

        local_storage.save_metadata(doc_id, file_ext, sample_metadata)

        metadata_file = local_storage._get_document_path(doc_id) / "metadata.json"
        assert metadata_file.exists()

        loaded = local_storage.load_metadata(doc_id, file_ext)
        assert loaded == sample_metadata
        assert loaded["filename"] == "test_policy.pdf"

    def test_save_and_load_document(self, local_storage, temp_document):
        """Test saving and retrieving the original document file."""
        doc_id = "test_doc_orig"
        file_ext = "pdf"

        local_storage.save_document(doc_id, temp_document, file_ext)

        saved_path = local_storage._get_document_path(doc_id) / f"document.{file_ext}"
        assert saved_path.exists()
        assert saved_path.read_text() == temp_document.read_text()

    def test_exists_returns_true_when_all_files_present(
        self, local_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test exists() returns True only when chunks + embeddings + metadata all present."""
        doc_id = "test_doc_full"
        file_ext = "pdf"

        assert not local_storage.exists(doc_id, file_ext)

        local_storage.save_chunks(doc_id, file_ext, sample_chunks)
        local_storage.save_embeddings(doc_id, file_ext, sample_embeddings)
        local_storage.save_metadata(doc_id, file_ext, sample_metadata)

        assert local_storage.exists(doc_id, file_ext)

    def test_exists_returns_false_when_partial_files(self, local_storage, sample_chunks):
        """Test exists() returns False when only some files are present."""
        doc_id = "test_doc_partial"
        file_ext = "pdf"

        local_storage.save_chunks(doc_id, file_ext, sample_chunks)
        assert not local_storage.exists(doc_id, file_ext)

    def test_delete_removes_all_files(
        self, local_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test delete() removes the entire document directory."""
        doc_id = "test_doc_delete"
        file_ext = "pdf"

        local_storage.save_chunks(doc_id, file_ext, sample_chunks)
        local_storage.save_embeddings(doc_id, file_ext, sample_embeddings)
        local_storage.save_metadata(doc_id, file_ext, sample_metadata)

        assert local_storage.exists(doc_id, file_ext)

        local_storage.delete(doc_id, file_ext)

        assert not local_storage.exists(doc_id, file_ext)
        assert not local_storage._get_document_path(doc_id).exists()

    def test_delete_nonexistent_document_no_error(self, local_storage):
        """Test deleting a document that doesn't exist does not raise."""
        local_storage.delete("nonexistent_doc_xyz", "pdf")

    def test_list_documents(
        self, local_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test listing all cached document IDs."""
        assert local_storage.list_documents() == []

        for i in range(3):
            doc_id = f"list_doc_{i}"
            local_storage.save_chunks(doc_id, "pdf", sample_chunks)
            local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
            local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        doc_list = local_storage.list_documents()
        assert len(doc_list) == 3
        assert "list_doc_0" in doc_list
        assert "list_doc_1" in doc_list
        assert "list_doc_2" in doc_list

    def test_get_stats(
        self, local_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test storage statistics calculation."""
        doc_id = "stats_doc"
        local_storage.save_chunks(doc_id, "pdf", sample_chunks)
        local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        stats = local_storage.get_stats()

        assert stats["backend"] == "local"
        assert stats["cache_dir"] == str(local_storage.cache_dir)
        assert stats["total_documents"] == 1
        assert stats["total_files"] == 3
        assert stats["total_size_mb"] > 0

    def test_delete_all(
        self, local_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test clearing the entire local cache."""
        for i in range(3):
            doc_id = f"clear_doc_{i}"
            local_storage.save_chunks(doc_id, "pdf", sample_chunks)
            local_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
            local_storage.save_metadata(doc_id, "pdf", sample_metadata)

        assert len(local_storage.list_documents()) == 3

        deleted = local_storage.delete_all()
        assert deleted == 3
        assert len(local_storage.list_documents()) == 0

    def test_load_missing_chunks_raises_file_not_found(self, local_storage):
        """Test that loading chunks from a missing document raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            local_storage.load_chunks("nonexistent_doc", "pdf")

    def test_load_missing_embeddings_raises_file_not_found(self, local_storage):
        """Test that loading embeddings from a missing document raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            local_storage.load_embeddings("nonexistent_doc", "pdf")

    def test_load_missing_metadata_raises_file_not_found(self, local_storage):
        """Test that loading metadata from a missing document raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            local_storage.load_metadata("nonexistent_doc", "pdf")


# ═══════════════════════════════════════════════════════════════════════
# Tests for S3StorageBackend (using moto for mocking)
# ═══════════════════════════════════════════════════════════════════════

class TestS3StorageBackend:
    """Tests for the S3 storage backend (mocked with moto)."""

    @pytest.fixture
    def s3_storage(self):
        """Create an S3StorageBackend with a mocked S3 service."""
        try:
            from moto import mock_aws
            import boto3
        except ImportError:
            pytest.skip("moto or boto3 not installed — skipping S3 tests")

        with mock_aws():
            s3_client = boto3.client("s3", region_name="us-east-1")
            s3_client.create_bucket(Bucket="idop-test-bucket")
            storage = S3StorageBackend(bucket_name="idop-test-bucket")
            yield storage

    def test_initialization(self, s3_storage):
        """Test S3StorageBackend initialization."""
        assert s3_storage.bucket_name == "idop-test-bucket"
        assert s3_storage.region == "us-east-1"

    def test_save_and_load_chunks(self, s3_storage, sample_chunks):
        """Test saving and loading chunks to S3."""
        doc_id = "s3_doc_chunks"
        file_ext = "pdf"

        s3_storage.save_chunks(doc_id, file_ext, sample_chunks)

        loaded = s3_storage.load_chunks(doc_id, file_ext)
        assert loaded == sample_chunks

    def test_save_and_load_embeddings(self, s3_storage, sample_embeddings):
        """Test saving and loading embeddings to S3."""
        doc_id = "s3_doc_emb"
        file_ext = "pdf"

        s3_storage.save_embeddings(doc_id, file_ext, sample_embeddings)

        loaded = s3_storage.load_embeddings(doc_id, file_ext)
        assert np.array_equal(loaded, sample_embeddings)

    def test_save_and_load_metadata(self, s3_storage, sample_metadata):
        """Test saving and loading metadata to S3."""
        doc_id = "s3_doc_meta"
        file_ext = "pdf"

        s3_storage.save_metadata(doc_id, file_ext, sample_metadata)

        loaded = s3_storage.load_metadata(doc_id, file_ext)
        assert loaded == sample_metadata

    def test_save_and_load_document(self, s3_storage, temp_document):
        """Test saving original document to S3."""
        doc_id = "s3_doc_orig"
        file_ext = "pdf"

        s3_storage.save_document(doc_id, temp_document, file_ext)

        key = s3_storage._get_s3_key(doc_id, file_ext, f"document.{file_ext}")
        assert s3_storage._object_exists(key)

    def test_exists_all_files(
        self, s3_storage, sample_chunks, sample_embeddings, sample_metadata, temp_document
    ):
        """Test exists() returns True when all S3 files are present."""
        doc_id = "s3_doc_full"
        file_ext = "pdf"

        assert not s3_storage.exists(doc_id, file_ext)

        s3_storage.save_document(doc_id, temp_document, file_ext)
        s3_storage.save_chunks(doc_id, file_ext, sample_chunks)
        s3_storage.save_embeddings(doc_id, file_ext, sample_embeddings)
        s3_storage.save_metadata(doc_id, file_ext, sample_metadata)

        assert s3_storage.exists(doc_id, file_ext)

    def test_exists_partial_files(self, s3_storage, sample_chunks):
        """Test exists() returns False when only some S3 files are present."""
        doc_id = "s3_doc_partial"
        file_ext = "pdf"

        s3_storage.save_chunks(doc_id, file_ext, sample_chunks)
        assert not s3_storage.exists(doc_id, file_ext)

    def test_delete(
        self, s3_storage, sample_chunks, sample_embeddings, sample_metadata, temp_document
    ):
        """Test deleting all S3 files for a document."""
        doc_id = "s3_doc_delete"
        file_ext = "pdf"

        s3_storage.save_document(doc_id, temp_document, file_ext)
        s3_storage.save_chunks(doc_id, file_ext, sample_chunks)
        s3_storage.save_embeddings(doc_id, file_ext, sample_embeddings)
        s3_storage.save_metadata(doc_id, file_ext, sample_metadata)

        assert s3_storage.exists(doc_id, file_ext)
        s3_storage.delete(doc_id, file_ext)
        assert not s3_storage.exists(doc_id, file_ext)

    def test_s3_key_structure(self, s3_storage, sample_chunks, temp_document):
        """Test that S3 keys follow the {doc_type}/{doc_id}/{filename} pattern."""
        doc_id = "s3_key_test"
        file_ext = "pdf"

        doc_key = s3_storage._get_s3_key(doc_id, file_ext, f"document.{file_ext}")
        chunks_key = s3_storage._get_s3_key(doc_id, file_ext, "chunks.json")

        assert doc_key == f"pdf/{doc_id}/document.pdf"
        assert chunks_key == f"pdf/{doc_id}/chunks.json"

    def test_list_documents(
        self, s3_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test listing all S3-cached documents."""
        for i in range(3):
            doc_id = f"s3_list_{i}"
            ext = "pdf" if i % 2 == 0 else "txt"
            s3_storage.save_chunks(doc_id, ext, sample_chunks)
            s3_storage.save_embeddings(doc_id, ext, sample_embeddings)
            s3_storage.save_metadata(doc_id, ext, sample_metadata)

        doc_list = s3_storage.list_documents()
        assert len(doc_list) == 3
        assert "s3_list_0" in doc_list
        assert "s3_list_1" in doc_list
        assert "s3_list_2" in doc_list

    def test_get_stats(
        self, s3_storage, sample_chunks, sample_embeddings, sample_metadata
    ):
        """Test S3 storage statistics."""
        doc_id = "s3_stats_doc"
        s3_storage.save_chunks(doc_id, "pdf", sample_chunks)
        s3_storage.save_embeddings(doc_id, "pdf", sample_embeddings)
        s3_storage.save_metadata(doc_id, "pdf", sample_metadata)

        stats = s3_storage.get_stats()

        assert stats["backend"] == "s3"
        assert stats["bucket"] == "idop-test-bucket"
        assert stats["region"] == "us-east-1"
        assert stats["total_documents"] == 1
        assert stats["total_objects"] >= 3
        assert stats["total_size_mb"] > 0
        assert "documents_by_type" in stats


# ═══════════════════════════════════════════════════════════════════════
# Backend Interface Compatibility Tests (parametrized)
# ═══════════════════════════════════════════════════════════════════════

class TestStorageBackendCompatibility:
    """Test that both backends implement the same interface correctly."""

    @pytest.fixture(params=["local", "s3"])
    def storage_backend(self, request, tmp_path):
        """Parametrized fixture to run the same tests on both backends."""
        if request.param == "local":
            yield LocalStorageBackend(cache_dir=tmp_path)
        elif request.param == "s3":
            try:
                from moto import mock_aws
                import boto3
            except ImportError:
                pytest.skip("moto or boto3 not installed — skipping S3 tests")

            with mock_aws():
                s3_client = boto3.client("s3", region_name="us-east-1")
                s3_client.create_bucket(Bucket="idop-test-bucket")
                yield S3StorageBackend(bucket_name="idop-test-bucket")

    def test_full_lifecycle(
        self, storage_backend, sample_chunks, sample_embeddings, sample_metadata, temp_document
    ):
        """Test save → exists → load → delete lifecycle on both backends."""
        doc_id = "compat_test_lifecycle"
        file_ext = "pdf"

        # Save
        storage_backend.save_document(doc_id, temp_document, file_ext)
        storage_backend.save_chunks(doc_id, file_ext, sample_chunks)
        storage_backend.save_embeddings(doc_id, file_ext, sample_embeddings)
        storage_backend.save_metadata(doc_id, file_ext, sample_metadata)

        # Exists
        assert storage_backend.exists(doc_id, file_ext)

        # Load and verify data integrity
        loaded_chunks = storage_backend.load_chunks(doc_id, file_ext)
        loaded_embeddings = storage_backend.load_embeddings(doc_id, file_ext)
        loaded_metadata = storage_backend.load_metadata(doc_id, file_ext)

        assert loaded_chunks == sample_chunks
        assert np.array_equal(loaded_embeddings, sample_embeddings)
        assert loaded_metadata == sample_metadata

        # Delete
        storage_backend.delete(doc_id, file_ext)
        assert not storage_backend.exists(doc_id, file_ext)
