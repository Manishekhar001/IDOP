from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path
import numpy as np


class StorageBackend(ABC):
    """
    Abstract interface for document cache storage.
    """

    @abstractmethod
    def exists(self, document_id: str, file_extension: str) -> bool:
        """
        Check if all cache files exist for a document.
        """
        pass

    @abstractmethod
    def save_document(
        self, document_id: str, file_path: Path, file_extension: str
    ) -> None:
        """
        Save original document file to storage.
        """
        pass

    @abstractmethod
    def save_chunks(
        self, document_id: str, file_extension: str, chunks: List[Dict]
    ) -> None:
        """
        Save chunks.json to storage.
        """
        pass

    @abstractmethod
    def save_embeddings(
        self, document_id: str, file_extension: str, embeddings: np.ndarray
    ) -> None:
        """
        Save embeddings.npy to storage.
        """
        pass

    @abstractmethod
    def save_metadata(
        self, document_id: str, file_extension: str, metadata: Dict
    ) -> None:
        """
        Save metadata.json to storage.
        """
        pass

    @abstractmethod
    def load_chunks(self, document_id: str, file_extension: str) -> List[Dict]:
        """
        Load chunks.json from storage.
        """
        pass

    @abstractmethod
    def load_embeddings(self, document_id: str, file_extension: str) -> np.ndarray:
        """
        Load embeddings.npy from storage.
        """
        pass

    @abstractmethod
    def load_metadata(self, document_id: str, file_extension: str) -> Dict:
        """
        Load metadata.json from storage.
        """
        pass

    @abstractmethod
    def delete(self, document_id: str, file_extension: str) -> None:
        """
        Delete all files for a document from storage.
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[str]:
        """
        List all cached document IDs.
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """
        Get storage statistics.
        """
        pass
