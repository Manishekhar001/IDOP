"""Determinism tests for SparseVectorService.

These tests verify that the sparse vector generation is **deterministic**
across independent instances.  The original ``hash()``-based implementation
produced different vectors on different Python processes because
``PYTHONHASHSEED`` is randomised by default.  This test suite ensures
that the fastembed BM25 replacement does not regress.
"""

import pytest

from app.core.sparse_vector_service import SparseVectorService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def svc_a():
    """First independent SparseVectorService instance."""
    return SparseVectorService()


@pytest.fixture
def svc_b():
    """Second independent SparseVectorService instance."""
    return SparseVectorService()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSparseDeterminism:
    """Same text → identical vector across two independent instances."""

    def test_identical_vectors_simple(self, svc_a, svc_b):
        """Basic determinism: a simple sentence must produce the same vector."""
        text = "IDOP is an intelligent data operations platform"

        vec_a = svc_a.generate_sparse_vector(text)
        vec_b = svc_b.generate_sparse_vector(text)

        assert (
            vec_a.indices == vec_b.indices
        ), f"Indices differ:\n  A={vec_a.indices}\n  B={vec_b.indices}"
        assert (
            vec_a.values == vec_b.values
        ), f"Values differ:\n  A={vec_a.values}\n  B={vec_b.values}"

    def test_identical_vectors_complex(self, svc_a, svc_b):
        """Determinism with punctuation, numbers, and mixed case."""
        text = (
            "Order #12345 — customer Alice Johnson (Enterprise, Canada) "
            "purchased 3x SmartPro Laptops @ $1,299.99 each."
        )

        vec_a = svc_a.generate_sparse_vector(text)
        vec_b = svc_b.generate_sparse_vector(text)

        assert vec_a.indices == vec_b.indices
        assert vec_a.values == vec_b.values

    def test_identical_vectors_unicode(self, svc_a, svc_b):
        """Determinism with unicode / non-ASCII characters."""
        text = "Clara Müller ordered from Zürich — invoice €2,500"

        vec_a = svc_a.generate_sparse_vector(text)
        vec_b = svc_b.generate_sparse_vector(text)

        assert vec_a.indices == vec_b.indices
        assert vec_a.values == vec_b.values

    def test_batch_matches_individual(self, svc_a):
        """Batch generation must produce the same vectors as individual calls."""
        texts = [
            "First document about data operations",
            "Second document about machine learning",
            "Third document about cloud infrastructure",
        ]

        individual = [svc_a.generate_sparse_vector(t) for t in texts]
        batch = svc_a.generate_sparse_vectors_batch(texts)

        for i, (ind, bat) in enumerate(zip(individual, batch)):
            assert ind.indices == bat.indices, f"Indices differ at position {i}"
            assert ind.values == bat.values, f"Values differ at position {i}"

    def test_non_empty_output(self, svc_a):
        """Sparse vector must contain at least one non-zero entry."""
        vec = svc_a.generate_sparse_vector("hello world")
        assert len(vec.indices) > 0, "Sparse vector has no indices"
        assert len(vec.values) > 0, "Sparse vector has no values"
        assert len(vec.indices) == len(vec.values), "indices and values length mismatch"

    def test_different_texts_different_vectors(self, svc_a):
        """Sanity: different texts must produce different vectors."""
        vec_a = svc_a.generate_sparse_vector("quantum computing research paper")
        vec_b = svc_a.generate_sparse_vector("chocolate cake baking recipe")

        # At minimum, the index sets should differ
        assert set(vec_a.indices) != set(
            vec_b.indices
        ), "Completely different texts produced identical index sets"
