import asyncio
import io
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from typing import Optional
from app.api.schemas import CollectionInfoResponse, DocumentUploadResponse, ErrorResponse
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


def get_vector_store(request: Request) -> VectorStoreService:
    return request.app.state.vector_store


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file format"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Upload a document",
)
async def upload_document(
    file: UploadFile = File(..., description="Document to upload (PDF, TXT, CSV)"),
    chunk_size: Optional[str] = Form(None, description="Custom target chunk size for parsing"),
    chunk_overlap: Optional[str] = Form(None, description="Custom chunk overlap"),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentUploadResponse:
    logger.info(f"Document upload: {file.filename} (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    # Safely parse chunk_size
    parsed_chunk_size = None
    if chunk_size is not None:
        val = str(chunk_size).strip()
        if val:
            try:
                parsed_chunk_size = int(val)
            except ValueError:
                raise HTTPException(status_code=400, detail="chunk_size must be an integer.")

    # Safely parse chunk_overlap
    parsed_chunk_overlap = None
    if chunk_overlap is not None:
        val = str(chunk_overlap).strip()
        if val:
            try:
                parsed_chunk_overlap = int(val)
            except ValueError:
                raise HTTPException(status_code=400, detail="chunk_overlap must be an integer.")

    try:
        file_bytes = await file.read()
        filename = file.filename

        loop = asyncio.get_event_loop()
        processor = DocumentProcessor(chunk_size=parsed_chunk_size, chunk_overlap=parsed_chunk_overlap)

        def _process():
            return processor.process_upload(io.BytesIO(file_bytes), filename)

        chunks = await loop.run_in_executor(None, _process)
        document_ids = await loop.run_in_executor(None, vector_store.add_documents, chunks)

        logger.info(f"Indexed {filename}: {len(chunks)} chunks, {len(document_ids)} IDs")
        return DocumentUploadResponse(
            message="Document uploaded and indexed successfully",
            filename=filename,
            chunks_created=len(chunks),
            document_ids=document_ids,
            chunk_size_applied=processor.chunk_size,
            chunk_overlap_applied=processor.chunk_overlap,
        )
    except ValueError as e:
        logger.warning(f"Invalid upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.get(
    "/info",
    response_model=CollectionInfoResponse,
    summary="Collection info",
)
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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve collection info: {str(e)}")


@router.delete(
    "/collection",
    summary="Delete collection",
)
async def delete_collection(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> dict:
    logger.warning("Collection deletion requested")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, vector_store.delete_collection)
        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Collection deletion error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete collection: {str(e)}")
