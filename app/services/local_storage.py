import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from app.services.storage_backend import StorageBackend
from app.config import get_settings

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """
    Filesystem-based storage for local development.
    """

    def __init__(self, cache_dir: Path = None):
        settings = get_settings()
        self.cache_dir = cache_dir or Path(
            settings.cache_dir
            if hasattr(settings, "cache_dir")
            else "data/cached_chunks"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorage initialized with cache_dir: {self.cache_dir}")

    def _get_document_path(self, document_id: str) -> Path:
        return self.cache_dir / document_id

    def exists(self, document_id: str, file_extension: str) -> bool:
        doc_path = self._get_document_path(document_id)
        required_files = [
            doc_path / "chunks.json",
            doc_path / "embeddings.npy",
            doc_path / "metadata.json",
        ]
        exist = all(f.exists() for f in required_files)
        if exist:
            logger.debug(f"Cache hit for document {document_id}")
        else:
            logger.debug(f"Cache miss for document {document_id}")
        return exist

    def save_document(
        self, document_id: str, file_path: Path, file_extension: str
    ) -> None:
        doc_path = self._get_document_path(document_id=document_id)
        doc_path.mkdir(parents=True, exist_ok=True)
        destination = doc_path / f"document.{file_extension}"
        shutil.copy2(file_path, destination)
        logger.info(f"Saved original document to {destination}")

    def save_chunks(
        self, document_id: str, file_extension: str, chunks: List[Dict]
    ) -> None:
        doc_path = self._get_document_path(document_id=document_id)
        doc_path.mkdir(parents=True, exist_ok=True)
        chunks_file = doc_path / "chunks.json"
        with open(chunks_file, "w") as f:
            json.dump(chunks, f, indent=2)
        logger.debug(f"Saved {len(chunks)} chunks to {chunks_file}")

    def save_embeddings(
        self, document_id: str, file_extension: str, embeddings: np.ndarray
    ) -> None:
        doc_path = self._get_document_path(document_id=document_id)
        doc_path.mkdir(parents=True, exist_ok=True)
        embeddings_file = doc_path / "embeddings.npy"
        np.save(embeddings_file, embeddings)
        logger.debug(f"Saved embeddings {embeddings.shape} to {embeddings_file}")

    def save_metadata(
        self, document_id: str, file_extension: str, metadata: Dict
    ) -> None:
        doc_path = self._get_document_path(document_id=document_id)
        doc_path.mkdir(parents=True, exist_ok=True)
        metadata_file = doc_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {metadata_file}")

    def load_chunks(self, document_id: str, file_extension: str) -> List[Dict]:
        chunks_file = self._get_document_path(document_id=document_id) / "chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        with open(chunks_file) as f:
            chunks = json.load(f)
        logger.debug(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks

    def load_embeddings(self, document_id: str, file_extension: str) -> np.ndarray:
        embeddings_file = (
            self._get_document_path(document_id=document_id) / "embeddings.npy"
        )
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        embeddings = np.load(embeddings_file)
        logger.debug(f"Loaded embeddings {embeddings.shape} from {embeddings_file}")
        return embeddings

    def load_metadata(self, document_id: str, file_extension: str) -> Dict:
        metadata_file = (
            self._get_document_path(document_id=document_id) / "metadata.json"
        )
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        with open(metadata_file) as f:
            metadata = json.load(f)
        logger.debug(f"Loaded metadata from {metadata_file}")
        return metadata

    def delete(self, document_id: str, file_extension: str) -> None:
        doc_path = self._get_document_path(document_id)
        if doc_path.exists():
            shutil.rmtree(doc_path)
            logger.info(f"Deleted cache for document {document_id}")
        else:
            logger.warning(f"Attempted to delete non-existent document {document_id}")

    def delete_all(self) -> int:
        count = 0
        if self.cache_dir.exists():
            for doc_dir in self.cache_dir.iterdir():
                if doc_dir.is_dir():
                    shutil.rmtree(doc_dir)
                    count += 1
        logger.info(f"Cleared entire local cache: {count} documents deleted")
        return count

    def list_documents(self) -> List[str]:
        if not self.cache_dir.exists():
            return []
        document_ids = [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
        logger.debug(f"Found {len(document_ids)} cached documents")
        return document_ids

    def get_stats(self) -> Dict:
        total_size = 0
        total_files = 0
        documents_count = 0
        if self.cache_dir.exists():
            for doc_dir in self.cache_dir.iterdir():
                if doc_dir.is_dir():
                    documents_count += 1
                    for file in doc_dir.iterdir():
                        if file.is_file():
                            total_size += file.stat().st_size
                            total_files += 1
        total_size_bytes = total_size
        if total_size_bytes < 1024:
            total_size_human = f"{total_size_bytes} B"
        elif total_size_bytes < 1024 * 1024:
            total_size_human = f"{total_size_bytes / 1024:.1f} KB"
        else:
            total_size_human = f"{total_size_bytes / (1024 * 1024):.2f} MB"

        stats = {
            "backend": "local",
            "cache_dir": str(self.cache_dir),
            "total_documents": documents_count,
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_bytes": total_size_bytes,
            "total_size_human": total_size_human,
        }
        logger.info(f"Local storage stats: {stats}")
        return stats
