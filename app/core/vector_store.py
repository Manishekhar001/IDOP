from functools import lru_cache
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

from app.config import get_settings
from app.core.embeddings import EmbeddingsService
from app.core.sparse_vector_service import SparseVectorService
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    logger.info(f"Connecting to Qdrant at: {settings.qdrant_url}")
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )
    logger.info("Qdrant client connected successfully")
    return client


class VectorStoreService:
    def __init__(self, collection_name: str | None = None) -> None:
        settings = get_settings()
        self.client = get_qdrant_client()
        self.collection_name = collection_name or settings.collection_name
        self.embeddings = EmbeddingsService()
        self.sparse_service = SparseVectorService()
        self.embedding_dimension = settings.embedding_dimension
        self._ensure_collection()
        logger.info(f"VectorStoreService ready — collection: {self.collection_name}")

    def _ensure_collection(self) -> None:
        """Create hybrid collection with dense and sparse vectors if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=self.embedding_dimension, distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )
                logger.info(f"Created hybrid collection: {self.collection_name}")
            else:
                logger.info(f"Collection '{self.collection_name}' exists")

            # Ensure payload indexes exist for duplicate checks and RAG filtering
            for field, schema in [
                ("content_hash", "keyword"),
                ("source", "keyword"),
                ("index", "integer"),
            ]:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                    logger.info(
                        f"Ensured Qdrant payload index for '{field}' ({schema})"
                    )
                except Exception as index_err:
                    logger.debug(
                        f"Payload index for '{field}' already exists or failed: {index_err}"
                    )
        except Exception as e:
            logger.error(f"Collection creation error: {e}")
            raise

    def _ensure_and_retry(self, func, *args, **kwargs):
        """Execute a Qdrant operation, ensuring the collection exists if it fails."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "not found" in err_str or "doesn't exist" in err_str or "404" in err_str:
                logger.warning(
                    f"Collection '{self.collection_name}' not found — recreating and retrying..."
                )
                self._ensure_collection()
                return func(*args, **kwargs)
            raise

    @track(name="vector_store_add_documents")
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Insert standard langchain Document items with dual vectors into Qdrant with SHA-256 chunk deduplication"""
        if not documents:
            logger.warning("add_documents called with empty list")
            return []

        import hashlib

        texts = [doc.page_content for doc in documents]
        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

        new_documents, new_texts, new_hashes, doc_ids = self._deduplicate_chunks(
            documents, texts, hashes
        )

        if not new_documents:
            logger.info(
                "All chunks already exist in Qdrant — skipped embedding entirely"
            )
            return doc_ids

        logger.info(
            f"Embedding {len(new_documents)}/{len(documents)} new chunks (skipped {len(documents) - len(new_documents)} duplicates)"
        )
        dense_embeddings = self.embeddings.embed_documents(new_texts)

        points, new_ids = self._build_points(
            new_documents, dense_embeddings, new_texts, new_hashes, doc_ids
        )
        doc_ids.extend(new_ids)

        try:
            self._ensure_and_retry(
                self.client.upsert, collection_name=self.collection_name, points=points
            )
            logger.info(
                f"Successfully upserted {len(points)} new chunks with dual vectors"
            )
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to upsert chunks: {e}")
            raise

    def _deduplicate_chunks(self, documents, texts, hashes):
        """Deduplicate chunks by content_hash against Qdrant."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        new_documents = []
        new_texts = []
        new_hashes = []
        doc_ids = []

        for doc, text, content_hash in zip(documents, texts, hashes):
            try:
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="content_hash", match=MatchValue(value=content_hash)
                        )
                    ]
                )
                existing = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=1,
                    with_payload=False,
                )[0]
            except Exception as e:
                logger.warning(f"Error checking duplicate chunk hash: {e}")
                existing = []

            if existing:
                logger.debug(f"Skipping duplicate chunk (hash={content_hash[:8]}...)")
                doc_ids.append(existing[0].id)
                continue

            doc.metadata["content_hash"] = content_hash
            new_documents.append(doc)
            new_texts.append(text)
            new_hashes.append(content_hash)

        return new_documents, new_texts, new_hashes, doc_ids

    def _build_points(self, documents, dense_embeddings, texts, hashes, doc_ids):
        """Build Qdrant PointStruct list from documents and embeddings.

        Returns (points, new_ids) tuple. The caller's doc_ids list is NOT
        mutated — new chunk IDs are returned separately to avoid implicit
        side-effect coupling.
        """
        from qdrant_client.models import PointStruct

        points = []
        new_ids = []
        for doc, dense, text, content_hash in zip(
            documents, dense_embeddings, texts, hashes
        ):
            chunk_id = str(uuid4())
            new_ids.append(chunk_id)

            sparse_vector = self.sparse_service.generate_sparse_vector(text)

            payload = {"content": text, **doc.metadata}

            points.append(
                PointStruct(
                    id=chunk_id,
                    vector={"dense": dense, "sparse": sparse_vector},
                    payload=payload,
                )
            )

        return points, new_ids

    @track(name="vector_store_add_with_embeddings")
    def add_documents_with_embeddings(
        self, documents: list[Document], dense_embeddings: list[list[float]]
    ) -> list[str]:
        """
        Insert documents with pre-computed dense embeddings (useful for cache-based uploads).
        Skips the embedding step and directly upserts to Qdrant with dual vectors.
        """
        if not documents:
            logger.warning("add_documents_with_embeddings called with empty list")
            return []

        if len(documents) != len(dense_embeddings):
            raise ValueError(
                f"Chunk/embedding mismatch: {len(documents)} chunks, {len(dense_embeddings)} embeddings"
            )

        import hashlib

        texts = [doc.page_content for doc in documents]
        hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]

        new_documents, new_texts, new_hashes, doc_ids = self._deduplicate_chunks(
            documents, texts, hashes
        )
        new_hashes_set = set(new_hashes)
        new_embeddings = [
            dense_embeddings[i] for i, h in enumerate(hashes) if h in new_hashes_set
        ]

        if not new_documents:
            logger.info("All chunks already exist in Qdrant — skipped upsert entirely")
            return doc_ids

        logger.info(
            f"Upserting {len(new_documents)}/{len(documents)} cached chunks (skipped {len(documents) - len(new_documents)} duplicates)"
        )

        points, new_ids = self._build_points(
            new_documents, new_embeddings, new_texts, new_hashes, doc_ids
        )
        doc_ids.extend(new_ids)

        try:
            self._ensure_and_retry(
                self.client.upsert, collection_name=self.collection_name, points=points
            )
            logger.info(
                f"Successfully upserted {len(points)} cached chunks with dual vectors"
            )
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to upsert cached chunks: {e}")
            raise

    def search_dense(
        self, query_vector: list[float], top_k: int, search_filter=None
    ) -> list:
        """Dense-only semantic search"""
        try:
            return self._ensure_and_retry(
                self.client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def search_sparse(self, query_text: str, top_k: int, search_filter=None) -> list:
        """Sparse-only keyword search (BM25)"""
        sparse_query = self.sparse_service.generate_sparse_vector(query_text)
        try:
            return self._ensure_and_retry(
                self.client.query_points,
                collection_name=self.collection_name,
                query=sparse_query,
                using="sparse",
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    def search_hybrid(
        self, query_vector: list[float], query_text: str, top_k: int, search_filter=None
    ) -> list:
        """Hybrid search with RRF fusion"""
        sparse_query = self.sparse_service.generate_sparse_vector(query_text)
        try:
            return self._ensure_and_retry(
                self.client.query_points,
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(query=sparse_query, using="sparse", limit=top_k * 3),
                    Prefetch(query=query_vector, using="dense", limit=top_k * 3),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k,
                with_payload=True,
            ).points
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    @track(name="vector_store_search")
    def search(
        self, query: str, k: int | None = None, mode: str = "hybrid"
    ) -> list[Document]:
        """Search method matching standard CSRAG RAG pipeline"""
        if not query:
            return []

        k = k or 4
        query_vector = self.embeddings.embed_query(query)

        try:
            if mode == "dense":
                results = self.search_dense(query_vector, k)
            elif mode == "sparse":
                results = self.search_sparse(query, k)
            else:
                results = self.search_hybrid(query_vector, query, k)

            documents = []
            for hit in results:
                content = hit.payload.get("content", "")
                metadata = {k: v for k, v in hit.payload.items() if k != "content"}
                metadata["score"] = hit.score
                documents.append(Document(page_content=content, metadata=metadata))

            return documents
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_chunk_by_index(self, source: str, index: int) -> Document | None:
        """Fetch a specific chronological chunk from the Qdrant database."""
        filter_cond = Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="index", match=MatchValue(value=index)),
            ]
        )
        try:
            res = self._ensure_and_retry(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=filter_cond,
                limit=1,
                with_payload=True,
            )[0]
            if res:
                hit = res[0]
                content = hit.payload.get("content", "")
                metadata = {k: v for k, v in hit.payload.items() if k != "content"}
                return Document(page_content=content, metadata=metadata)
            return None
        except Exception as e:
            logger.error(f"Failed to fetch chunk by index: {e}")
            return None

    @track(name="vector_store_delete_collection")
    def delete_collection(self) -> None:
        logger.warning(f"Deleting collection: {self.collection_name}")
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
        self._ensure_collection()

    def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status.value,
            }
        except Exception:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "status": "not_found",
            }

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
