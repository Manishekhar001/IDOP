import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.services.local_storage import LocalStorageBackend
from app.services.s3_storage import S3StorageBackend
from app.services.storage_backend import StorageBackend

logger = logging.getLogger("idop_app.cache_service")


class CacheService:
    """
    Manages caching of document chunks and embeddings.
    """

    def __init__(self, storage_backend: StorageBackend | None = None):
        self.init_error: str | None = None
        settings = get_settings()
        if storage_backend is None:
            backend_type = getattr(settings, "storage_backend", "s3").lower()

            if backend_type == "s3":
                try:
                    self.storage = S3StorageBackend()
                    if not self.storage.enabled:
                        # Capture the actual S3 validation error for diagnostics
                        s3_err = getattr(self.storage, "validation_error", None)
                        err_msg = (
                            f"S3 disabled: {s3_err}"
                            if s3_err
                            else (
                                f"S3 storage initialized but reported disabled. "
                                f"Bucket: '{settings.s3_cache_bucket}', Region: {settings.aws_region}. "
                                f"Check bucket existence, IAM permissions (s3:HeadBucket, s3:PutObject, s3:GetObject), "
                                f"and that the bucket is in region {settings.aws_region}."
                            )
                        )
                        self.init_error = err_msg
                        logger.critical(err_msg)
                        self.storage = LocalStorageBackend()
                    else:
                        logger.info(
                            f"Using S3 storage (bucket: {settings.s3_cache_bucket})"
                        )
                except Exception as e:
                    err_msg = (
                        f"Failed to initialize S3 storage: {e}. "
                        f"Bucket: '{settings.s3_cache_bucket}', Region: {settings.aws_region}. "
                        f"Check bucket name, IAM permissions, and network connectivity."
                    )
                    self.init_error = err_msg
                    logger.critical(err_msg)
                    self.storage = LocalStorageBackend()
            elif backend_type == "local":
                self.storage = LocalStorageBackend()
                cache_dir_val = getattr(settings, "cache_dir", "data/cached_chunks")
                logger.info(f"Using local storage (dir: {cache_dir_val})")
            else:
                raise ValueError(f"Unknown storage backend: {backend_type}")
        else:
            self.storage = storage_backend
            logger.info(
                f"Using custom storage backend: {type(storage_backend).__name__}"
            )

    def compute_document_id(self, file_path: Path) -> str:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        doc_id = sha256.hexdigest()
        logger.debug(f"Computed document ID: {doc_id} for {file_path.name}")
        return doc_id

    def cache_exists(self, doc_id: str, file_extension: str) -> bool:
        try:
            return self.storage.exists(doc_id, file_extension)
        except Exception as e:
            logger.warning(f"Error checking cache for {doc_id}: {e}")
            return False

    def save_document(self, doc_id: str, file_path: str, file_extension: str) -> None:
        try:
            self.storage.save_document(doc_id, Path(file_path), file_extension)
            logger.info(f"Saved original document {doc_id}.{file_extension}")
        except Exception as e:
            logger.error(f"Failed to save document {doc_id}: {e}")
            raise

    def save_chunks_and_embeddings(
        self,
        doc_id: str,
        file_extension: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)

            self.storage.save_chunks(doc_id, file_extension, chunks)
            self.storage.save_embeddings(doc_id, file_extension, embeddings_array)
            self.storage.save_metadata(doc_id, file_extension, metadata)

            logger.info(
                f"Cached {len(chunks)} chunks for {doc_id} (type: {file_extension})"
            )

        except Exception as e:
            logger.error(f"Failed to cache document {doc_id}: {e}")
            try:
                self.storage.delete(doc_id, file_extension)
            except Exception:
                pass
            raise Exception(f"Failed to save cache: {e!s}")

    def load_chunks_and_embeddings(
        self, doc_id: str, file_extension: str
    ) -> dict[str, Any] | None:
        if not self.cache_exists(doc_id, file_extension):
            return None

        try:
            chunks = self.storage.load_chunks(doc_id, file_extension)
            embeddings_array = self.storage.load_embeddings(doc_id, file_extension)
            metadata = self.storage.load_metadata(doc_id, file_extension)

            embeddings = embeddings_array.tolist()

            if len(chunks) != len(embeddings):
                logger.error(
                    f"Cache corruption: {len(chunks)} chunks but {len(embeddings)} embeddings."
                )
                return None

            logger.info(f"Loaded {len(chunks)} chunks from cache for {doc_id}")
            return {"chunks": chunks, "embeddings": embeddings, "metadata": metadata}

        except Exception as e:
            logger.warning(f"Failed to load cache for {doc_id}: {e!s}")
            return None

    def get_cache_stats(self) -> dict[str, Any]:
        try:
            return self.storage.get_stats()
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e), "total_documents": 0}

    def clear_cache(
        self, doc_id: str | None = None, file_extension: str | None = None
    ) -> dict[str, Any]:
        try:
            if doc_id:
                if not file_extension:
                    return {
                        "cleared": False,
                        "message": "file_extension required when clearing specific document",
                        "documents_cleared": 0,
                    }
                self.storage.delete(doc_id, file_extension)
                logger.info(f"Cleared cache for document: {doc_id}")
                return {
                    "cleared": True,
                    "message": f"Cleared cache for document {doc_id}",
                    "documents_cleared": 1,
                }
            else:
                if hasattr(self.storage, "delete_all"):
                    objects_deleted = self.storage.delete_all()
                    logger.info(
                        f"Cleared entire cache: {objects_deleted} objects deleted"
                    )
                    return {
                        "cleared": True,
                        "message": f"Cleared entire cache ({objects_deleted} objects deleted)",
                        "documents_cleared": "all",
                        "objects_deleted": objects_deleted,
                    }
                else:
                    # Fallback if no delete_all
                    doc_ids = self.storage.list_documents()
                    for did in doc_ids:
                        self.storage.delete(did, "pdf")  # best-effort deletion
                    return {
                        "cleared": True,
                        "message": "Cleared entire cache (best effort)",
                        "documents_cleared": len(doc_ids),
                    }
        except Exception as e:
            logger.error(f"Failed to clear cache: {e!s}")
            return {
                "cleared": False,
                "message": f"Failed to clear cache: {e!s}",
                "documents_cleared": 0,
            }
