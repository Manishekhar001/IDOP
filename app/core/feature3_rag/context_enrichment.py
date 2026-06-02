import logging
from typing import List
from langchain_core.documents import Document
from app.core.vector_store import VectorStoreService
from app.opik import track

logger = logging.getLogger("idop_app.context_enrichment")


class ContextEnrichmentService:
    """
    Implements Context Enrichment Window. Fetches neighboring chunks
    chronologically to expand the context window around each retrieved chunk.

    Accepts an optional shared VectorStoreService to avoid creating a new
    Qdrant connection on every enrichment. Falls back to creating one if
    none is provided.
    """

    def __init__(self, vector_store: VectorStoreService | None = None):
        self.vector_store = vector_store or VectorStoreService()

    @track(name="context_enrichment_enrich")
    def enrich_documents(
        self, documents: List[Document], num_neighbors: int = 1, chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Enrich retrieved documents with their chronological neighbors.
        """
        if not documents:
            return []

        logger.info(
            f"Enriching {len(documents)} retrieved documents (num_neighbors={num_neighbors})"
        )
        enriched_documents = []

        for doc in documents:
            current_index = doc.metadata.get("index")
            source = doc.metadata.get("source") or doc.metadata.get("source_file")

            if current_index is None or not source:
                # Can't enrich without index or source info, keep original
                enriched_documents.append(doc)
                continue

            # Calculate bounds
            start_index = max(0, current_index - num_neighbors)
            end_index = current_index + num_neighbors + 1

            # Fetch neighbors
            neighbor_chunks = []
            for idx in range(start_index, end_index):
                if idx == current_index:
                    neighbor_chunks.append(doc)
                else:
                    neighbor_chunk = self.vector_store.get_chunk_by_index(source, idx)
                    if neighbor_chunk:
                        neighbor_chunks.append(neighbor_chunk)

            # Sort neighbor chunks by index
            neighbor_chunks.sort(key=lambda x: x.metadata.get("index", 0))

            if not neighbor_chunks:
                enriched_documents.append(doc)
                continue

            # Concatenate chunks accounting for overlap
            concatenated_text = neighbor_chunks[0].page_content
            for i in range(1, len(neighbor_chunks)):
                current_chunk = neighbor_chunks[i].page_content
                # Safe overlap concatenation
                overlap_start = max(0, len(concatenated_text) - chunk_overlap)
                concatenated_text = concatenated_text[:overlap_start] + current_chunk

            # Create enriched document preserving scores and original metadata
            enriched_doc = Document(
                page_content=concatenated_text,
                metadata={
                    **doc.metadata,
                    "enriched": True,
                    "original_content": doc.page_content,
                },
            )
            enriched_documents.append(enriched_doc)

        logger.info(
            f"✓ Context Enrichment complete. Enriched {len(enriched_documents)} documents."
        )
        return enriched_documents
