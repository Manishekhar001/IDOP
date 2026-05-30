import logging
from typing import List
from langchain_core.documents import Document
from app.core.vector_store import VectorStoreService
from app.core.embeddings import get_embeddings
from app.core.feature3_rag.hyde import HydeService
from app.config import get_settings
from app.opik import track

logger = logging.getLogger("idop_app.retrieval")


class RetrievalService:
    def __init__(self):
        settings = get_settings()
        self.vector_store = VectorStoreService()
        self.embeddings = get_embeddings()
        self.hyde_service = HydeService()
        self.settings = settings

    @track(name="retrieval_service_retrieve")
    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        use_hyde: bool = False,
        search_mode: str = "hybrid",
    ) -> List[Document]:
        """
        Retrieves relevant documents with optional HyDE expansion.
        """
        if use_hyde:
            # Generate hypothetical documents
            hypotheses = self.hyde_service.generate_hypothetical_documents(query)

            # Search vector store for each hypothesis
            all_results = []
            for hypothesis in hypotheses:
                results = self.vector_store.search(
                    query=hypothesis, k=top_k, mode=search_mode
                )
                all_results.extend(results)

            # Deduplicate by content
            unique_docs = self._merge_and_deduplicate(all_results, top_k)
            logger.info(f"HyDE: Retrieved {len(unique_docs)} unique documents")
            return unique_docs
        else:
            # Standard retrieval
            results = self.vector_store.search(query=query, k=top_k, mode=search_mode)
            logger.info(f"Standard retrieval: Retrieved {len(results)} documents")
            return results

    def _merge_and_deduplicate(
        self, all_docs: List[Document], top_k: int
    ) -> List[Document]:
        seen_content = set()
        deduplicated = []

        # Sort by score descending (Qdrant search includes 'score' in metadata or returns it)
        sorted_docs = sorted(
            all_docs, key=lambda x: x.metadata.get("score", 0.0), reverse=True
        )

        for doc in sorted_docs:
            content = doc.page_content.strip()
            if content not in seen_content:
                seen_content.add(content)
                deduplicated.append(doc)

        return deduplicated[:top_k]
