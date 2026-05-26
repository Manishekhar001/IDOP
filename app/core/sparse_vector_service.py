import re
from collections import Counter
from typing import List
from qdrant_client.models import SparseVector


class SparseVectorService:
    """Service for generating sparse vectors for BM25-style search."""

    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
        'what', 'when', 'where', 'who', 'which', 'why', 'how', 'or', 'if',
        'each', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now'
    }

    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+(?:-[a-z0-9]+)*\b', text)
        tokens = [t for t in tokens if t not in self.STOP_WORDS]
        return tokens

    def _hash_token(self, token: str) -> int:
        return abs(hash(token)) % (2**32)

    def generate_sparse_vector(self, text: str) -> SparseVector:
        tokens = self.tokenize(text)
        term_frequencies = Counter(tokens)

        indices = []
        values = []

        for token, freq in term_frequencies.items():
            index = self._hash_token(token)
            indices.append(index)
            values.append(float(freq))

        return SparseVector(indices=indices, values=values)

    def generate_sparse_vectors_batch(self, texts: List[str]) -> List[SparseVector]:
        return [self.generate_sparse_vector(text) for text in texts]
