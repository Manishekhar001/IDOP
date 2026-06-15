import asyncio
import hashlib
import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.schemas import (
    CollectionInfoResponse,
    DocumentUploadResponse,
    ErrorResponse,
)
from app.core.document_processor import DocumentProcessor
from app.core.embeddings import EmbeddingQuotaError
from app.core.vector_store import VectorStoreService
from app.opik import track
from app.services.cache_init import get_doc_cache
from app.utils.logger import get_logger
from app.utils.validators import FileValidator, ValidationError

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


def get_vector_store(request: Request) -> VectorStoreService:
    return request.app.state.vector_store


def _get_extension(filename: str) -> str:
    """Extract the file extension without the leading dot."""
    _, ext = os.path.splitext(filename)
    return ext.lstrip(".") if ext else "bin"


def _validate_file_size(file_bytes: bytes) -> None:
    """Validate file size against the configured maximum."""
    max_size = FileValidator.MAX_FILE_SIZE  # 50MB by default
    if len(file_bytes) > max_size:
        max_mb = max_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds maximum allowed size of {max_mb:.0f} MB",
        )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file format"},
        413: {"description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Upload a document for processing",
    description=(
        "Upload and process a document (PDF, TXT, CSV) through the ingestion pipeline. "
        "Pipeline: validate → cache check → parse → chunk → embed → index in Qdrant → cache. "
        "Returns document IDs and chunk metadata for downstream querying."
    ),
)
@track(name="upload_document")
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (PDF, TXT, CSV)"),
    chunk_size: int | None = Form(
        None, description="Custom target chunk size for parsing"
    ),
    chunk_overlap: int | None = Form(None, description="Custom chunk overlap"),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """
    Upload and process a document through the ingestion pipeline.

    Pipeline: validate → cache check (SHA-256) → parse → chunk → embed →
    index in Qdrant → cache chunks + embeddings for future cache hits.

    Args:
        file: The document file to upload (PDF, TXT, CSV). Max 50 MB.
        chunk_size: Optional custom chunk size override.
        chunk_overlap: Optional custom chunk overlap override.
        vector_store: Injected Qdrant vector store dependency.

    Returns:
        DocumentUploadResponse: Upload status with filename, chunk count, document IDs,
                               chunking parameters applied, and cache hit indicator.

    Raises:
        HTTPException 400: If the file format is unsupported or no text can be extracted.
        HTTPException 413: If the file exceeds the maximum allowed size.
        HTTPException 500: If embedding, indexing, or cache operations fail.
    """
    logger.info(
        f"Document upload: {file.filename} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
    )

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    # Validate file extension early using FileValidator
    try:
        FileValidator.validate_file(file)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # FastAPI handles int parsing and validation via the Optional[int] type hint
    parsed_chunk_size = chunk_size
    parsed_chunk_overlap = chunk_overlap

    try:
        file_bytes = await file.read()
        filename = file.filename
        file_ext = _get_extension(filename)

        # Validate file size before any processing
        _validate_file_size(file_bytes)

        # Compute document ID (SHA-256 of file content)
        doc_id = hashlib.sha256(file_bytes).hexdigest()
        logger.info(
            f"Computed doc_id={doc_id[:12]}... for {filename} ({len(file_bytes) / 1024:.1f} KB)"
        )

        loop = asyncio.get_running_loop()

        # --- Cache Check ---
        doc_cache = get_doc_cache()
        cache_hit = False
        cached = None
        if doc_cache is not None:
            try:
                cache_hit = doc_cache.cache_exists(doc_id, file_ext)
            except Exception as e:
                logger.warning(f"Cache exists check failed: {e}")
        if cache_hit:
            logger.info(f"Cache HIT for {filename} (doc_id={doc_id[:12]}...)")
            try:
                cached = doc_cache.load_chunks_and_embeddings(doc_id, file_ext)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
            if cached is not None:
                embeddings = cached["embeddings"]
                # Validate cached embedding dimension matches current config
                # Prevents dimension mismatch (e.g., old OpenAI 1536-dim vs Nomic 768-dim)
                expected_dim = vector_store.embedding_dimension
                if (
                    embeddings
                    and len(embeddings) > 0
                    and len(embeddings[0]) != expected_dim
                ):
                    logger.warning(
                        f"Cached embedding dimension ({len(embeddings[0])}) doesn't match "
                        f"expected ({expected_dim}). Ignoring cache and re-embedding..."
                    )
                else:
                    # Reconstruct Document objects from cached chunks
                    chunks = [
                        DocumentProcessor._dict_to_document(chunk)
                        for chunk in cached["chunks"]
                    ]

                    document_ids = await loop.run_in_executor(
                        None,
                        vector_store.add_documents_with_embeddings,
                        chunks,
                        embeddings,
                    )

                    logger.info(
                        f"Cache HIT - Indexed {filename}: {len(chunks)} chunks (from cache), {len(document_ids)} IDs"
                    )
                    return DocumentUploadResponse(
                        message="Document uploaded and indexed successfully (from cache)",
                        filename=filename,
                        chunks_created=len(chunks),
                        document_ids=document_ids,
                        chunk_size_applied=cached["metadata"].get(
                            "chunk_size", parsed_chunk_size
                        ),
                        chunk_overlap_applied=cached["metadata"].get(
                            "chunk_overlap", parsed_chunk_overlap
                        ),
                        cache_hit=True,
                    )
            else:
                logger.warning(
                    f"Cache load returned None for doc_id={doc_id[:12]}... despite cache_exists=True — re-processing"
                )

        # --- Cache MISS: Process document normally ---
        logger.info(
            f"Cache MISS for {filename} (doc_id={doc_id[:12]}...) — processing from scratch"
        )

        processor = DocumentProcessor(
            chunk_size=parsed_chunk_size, chunk_overlap=parsed_chunk_overlap
        )

        # Process file from bytes directly (avoids redundant BytesIO wrapper)
        def _process():
            return processor.process_upload_bytes(file_bytes, filename)

        chunks = await loop.run_in_executor(None, _process)

        # Compute embeddings ONCE for both Qdrant upsert and cache storage
        texts = [doc.page_content for doc in chunks]
        if not texts:
            raise HTTPException(
                status_code=400,
                detail="No text content could be extracted from the document.",
            )

        logger.info(
            f"Embedding {len(chunks)} chunks ({sum(len(t) for t in texts)} chars)..."
        )
        embeddings = await loop.run_in_executor(
            None, vector_store.embeddings.embed_documents, texts
        )

        # Upsert to Qdrant with the pre-computed embeddings
        document_ids = await loop.run_in_executor(
            None, vector_store.add_documents_with_embeddings, chunks, embeddings
        )

        # --- Save to cache for future requests ---
        if doc_cache is not None:
            try:
                chunk_dicts = [
                    {"text": doc.page_content, "metadata": doc.metadata}
                    for doc in chunks
                ]
                metadata = {
                    "filename": filename,
                    "chunk_size": processor.chunk_size,
                    "chunk_overlap": processor.chunk_overlap,
                    "total_chunks": len(chunks),
                }

                await loop.run_in_executor(
                    None,
                    doc_cache.save_chunks_and_embeddings,
                    doc_id,
                    file_ext,
                    chunk_dicts,
                    embeddings,
                    metadata,
                )
                logger.info(f"Cached {len(chunks)} chunks for {filename}")
            except Exception as e:
                logger.warning(f"Failed to cache chunks for {filename}: {e}")

        logger.info(
            f"Indexed {filename}: {len(chunks)} chunks, {len(document_ids)} IDs"
        )
        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            filename=filename,
            chunks_created=len(chunks),
            document_ids=document_ids,
            chunk_size_applied=processor.chunk_size,
            chunk_overlap_applied=processor.chunk_overlap,
            cache_hit=False,
        )
    except ValueError as e:
        logger.warning(f"Invalid upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        # Distinguish OpenAI quota errors (429) from generic 500 failures
        if isinstance(e, EmbeddingQuotaError):
            logger.error(f"OpenAI quota exhausted during document upload: {e}")
            raise HTTPException(
                status_code=429,
                detail=str(e),
            )
        # Catch raw openai.RateLimitError as a safety net for any non-embedding routes
        err_str = str(e).lower()
        if "429" in err_str or "insufficient_quota" in err_str:
            logger.error(f"OpenAI quota error during document upload: {e}")
            raise HTTPException(
                status_code=429,
                detail=(
                    "OpenAI API quota exhausted. The API key needs more credits.\n"
                    "To resolve this:\n"
                    "1. Visit https://platform.openai.com/account/billing to add credits\n"
                    "2. Or set a new OPENAI_API_KEY with available quota in the deployment secrets"
                ),
            )
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {e!s}"
        )


@router.get(
    "/info",
    response_model=CollectionInfoResponse,
    summary="Get Qdrant collection information",
    description="Returns the current Qdrant vector store collection metadata including name, total indexed documents, and cluster status.",
)
@track(name="collection_info")
async def collection_info(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> CollectionInfoResponse:
    try:
        info = vector_store.get_collection_info()
        return CollectionInfoResponse(
            collection_name=info["name"],
            total_documents=info["points_count"],
            status=info["status"],
        )
    except Exception as e:
        logger.error(f"Collection info error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve collection info: {e!s}"
        )


@router.delete(
    "/collection",
    summary="Delete the entire Qdrant collection",
    description="Danger operation — permanently removes the entire Qdrant vector collection and all indexed documents. Use with caution.",
)
@track(name="delete_collection")
async def delete_collection(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict:
    logger.warning("Collection deletion requested")
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, vector_store.delete_collection)
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Collection deletion error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete collection: {e!s}"
        )
