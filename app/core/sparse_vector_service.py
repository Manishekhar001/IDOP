"""Sparse vector generation via fastembed Qdrant/BM25.

Delegates to ``fastembed.SparseTextEmbedding`` with the ``Qdrant/bm25``
model.  This replaces the previous ``hash()``-based approach which was
**non-deterministic** across Python processes (``PYTHONHASHSEED``).

The BM25 model is a statistical model (IDF lookup + tokenization), *not*
a neural forward pass, so memory footprint is minimal (~30-50 MB).
"""

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector


class SparseVectorService:
    """Service for generating sparse vectors for BM25-style search."""

    # Default model — Qdrant's BM25 implementation via fastembed
    MODEL_NAME = "Qdrant/bm25"

    def __init__(self, model_name: str | None = None) -> None:
        self._model = SparseTextEmbedding(model_name=model_name or self.MODEL_NAME)

    def generate_sparse_vector(self, text: str) -> SparseVector:
        """Generate a sparse vector for a single text.

        Returns a ``qdrant_client.models.SparseVector`` — the same type
        the old implementation returned, so callers need no changes.
        """
        # embed() returns a generator; we only need the first (and only) result
        embedding = next(self._model.embed([text]))
        return SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )

    def generate_sparse_vectors_batch(self, texts: list[str]) -> list[SparseVector]:
        """Generate sparse vectors for a batch of texts."""
        return [
            SparseVector(
                indices=embedding.indices.tolist(), values=embedding.values.tolist()
            )
            for embedding in self._model.embed(texts)
        ]
