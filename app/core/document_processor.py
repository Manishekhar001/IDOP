import tempfile
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.opik import track
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".csv"}

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        logger.info(
            f"DocumentProcessor ready — "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

    def _load_pdf(self, file_path: Path) -> list[Document]:
        """
        Extract text from a PDF using pypdf.

        Uses pypdf.PdfReader for lightweight text extraction from born-digital
        (text-based) PDFs. Does NOT use torch, OCR, or ML models — eliminates
        OOM kills on t2.micro (1 GB RAM) which occurred with docling's torch-
        based layout detection models.

        NOTE: pypdf extracts raw text in reading order. Tables with visible
        borders are extracted row-by-row but column alignment is not preserved.
        For structured table extraction, consider replacing with pdfplumber.

        DOCLING CODE DELETED (2026-06-07): The previous implementation used
        docling's DocumentConverter + HybridChunker for ML-powered layout
        analysis. It was removed because torch + docling consumed ~1 GB RAM
        causing OOM kills on t2.micro (1 GB). See git history for the
        original implementation.
        """
        logger.info(f"Loading PDF with pypdf: {file_path.name}")
        try:
            import pypdf
        except ImportError as e:
            raise ImportError(
                f"pypdf not available: {e}. Install with: pip install pypdf>=3.0.0"
            ) from e

        reader = pypdf.PdfReader(str(file_path))
        all_text: list[str] = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                all_text.append(text)

        if not all_text:
            logger.warning(f"No text extracted from {file_path.name} — may be a scanned PDF")
            return []

        full_text = "\n\n".join(all_text)
        doc = Document(
            page_content=full_text,
            metadata={
                "source": file_path.name,
                "pages": len(reader.pages),
            },
        )
        logger.info(
            f"Loaded {file_path.name}: {len(reader.pages)} pages, "
            f"{len(full_text)} chars extracted"
        )
        return [doc]

    def _load_text(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading text: {file_path.name}")
        text = file_path.read_text(encoding="utf-8")
        docs = [Document(page_content=text, metadata={"source": file_path.name})]
        logger.info(f"Loaded {file_path.name}")
        return docs

    def _load_csv(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading CSV: {file_path.name}")
        df = pd.read_csv(file_path, encoding="utf-8")
        docs = []
        for _, row in df.iterrows():
            content = " | ".join(f"{col}: {val}" for col, val in row.items())
            docs.append(
                Document(page_content=content, metadata={"source": file_path.name})
            )
        logger.info(f"Loaded {len(docs)} rows from {file_path.name}")
        return docs

    def load_file(self, file_path: str | Path) -> list[Document]:
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".csv": self._load_csv,
        }
        return loaders[ext](file_path)

    def load_from_upload(self, file: BinaryIO, filename: str) -> list[Document]:
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            docs = self.load_file(tmp_path)
            for doc in docs:
                doc.metadata["source"] = filename
            return docs
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = self.text_splitter.split_documents(documents)

        # Tag each chunk with a chronological index and source for Context Enrichment
        for idx, chunk in enumerate(chunks):
            chunk.metadata["index"] = idx
            # Ensure "source" is set in case it's missing
            if "source" not in chunk.metadata:
                chunk.metadata["source"] = chunk.metadata.get(
                    "source_file", "unknown_source"
                )

        logger.info(f"Created {len(chunks)} chunks with index tracking metadata")
        return chunks

    @staticmethod
    def _dict_to_document(chunk_dict: dict) -> Document:
        """
        Reconstruct a langchain Document from a cached chunk dictionary.

        The dictionary is expected to have 'text' and 'metadata' keys,
        as stored by CacheService.save_chunks_and_embeddings.
        """
        return Document(
            page_content=chunk_dict.get("text", ""),
            metadata=chunk_dict.get("metadata", {}),
        )

    def process_upload(self, file: BinaryIO, filename: str) -> list[Document]:
        """Process a file upload from a BinaryIO stream.

        All file types (PDF, TXT, CSV) are split using the text splitter.
        PDFs are no longer pre-chunked by docling's HybridChunker — pypdf
        returns raw full-page text, so splitting is required for all types.
        """
        docs = self.load_from_upload(file, filename)
        return self.split_documents(docs)

    @track(name="document_processor_process")
    def process_upload_bytes(self, file_bytes: bytes, filename: str) -> list[Document]:
        """
        Process a file upload from raw bytes, avoiding redundant BytesIO wrapping.
        Writes to a single temp file for the underlying loader to read.

        All file types (PDF, TXT, CSV) are split using the text splitter.
        PDFs are no longer pre-chunked by docling's HybridChunker — pypdf
        returns raw full-page text, so splitting is required for all types.
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            docs = self.load_file(tmp_path)
            for doc in docs:
                doc.metadata["source"] = filename
            return self.split_documents(docs)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
