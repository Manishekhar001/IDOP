import logging
from typing import List
from langchain_core.documents import Document
from app.opik import track
from app.config import get_settings

logger = logging.getLogger("idop_app.reranking")


class RerankingService:
    """
    Reranking service using Voyage AI API exclusively, with soft fallback.
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.voyage_api_key
        self.model = "rerank-2.5"
        self.enabled = False

        if self.api_key:
            try:
                import voyageai

                self.client = voyageai.Client(api_key=self.api_key)
                self.enabled = True
                logger.info("Voyage AI reranking service initialized successfully")
            except ImportError:
                logger.warning(
                    "voyageai package not installed. Reranking will be bypassed."
                )
        else:
            logger.info("VOYAGE_API_KEY not configured. Reranking will be bypassed.")

    @track(name="reranking_service_rerank")
    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """
        Rerank a list of documents relative to the user query.
        """
        if not documents:
            return []

        if not self.enabled:
            # Fallback to returning original top_k sorted by score metadata if present
            logger.info("Reranking disabled. Returning top_k of original list.")
            return documents[:top_k]

        try:
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

        except Exception as e:
            logger.error(f"Voyage AI Reranking failed: {e}. Falling back.")
            return documents[:top_k]
