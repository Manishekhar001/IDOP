import logging
from typing import List
from langchain_core.documents import Document
from app.opik import track
from app.config import get_settings

logger = logging.getLogger("idop_app.reranking")


class RerankingService:
    """
    Reranking service using Voyage AI exclusively.

    Raises on initialization failure (missing package or API key) so callers
    know immediately whether reranking is available. The rerank() method
    propagates API errors directly — no silent fallback to original order.
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.voyage_api_key
        self.model = "rerank-2.5"

        if not self.api_key:
            raise ValueError(
                "VOYAGE_API_KEY is required for reranking but is not configured."
            )

        try:
            import voyageai
        except ImportError:
            raise ImportError(
                "voyageai package is not installed. Install it with: pip install voyageai"
            ) from None

        self.client = voyageai.Client(api_key=self.api_key)
        logger.info("Voyage AI reranking service initialized successfully")

    @track(name="reranking_service_rerank")
    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """
        Rerank a list of documents relative to the user query using Voyage AI.
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        logger.info(f"Reranking {len(texts)} documents using Voyage AI")

        response = self.client.rerank(
            query=query, documents=texts, model=self.model, top_k=top_k
        )

        reranked_docs = []
        for result in response.results:
            original_doc = documents[result.index]
            original_doc.metadata["rerank_score"] = float(result.relevance_score)
            reranked_docs.append(original_doc)

        logger.info(
            f"Reranking complete. Returned {len(reranked_docs)} reranked documents."
        )
        return reranked_docs
