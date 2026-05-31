import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import CSVLoader, TextLoader
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
        logger.info(f"Loading PDF with Docling: {file_path.name}")
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))

        # Export to markdown to preserve document structure (headings, tables, lists)
        text = result.document.export_to_markdown()

        doc = Document(
            page_content=text,
            metadata={"source": file_path.name},
        )
        logger.info(
            f"Loaded {len(text)} characters from {file_path.name} using Docling"
        )
        return [doc]

    def _load_text(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading text: {file_path.name}")
        docs = TextLoader(str(file_path), encoding="utf-8").load()
        logger.info(f"Loaded {file_path.name}")
        return docs

    def _load_csv(self, file_path: Path) -> list[Document]:
        logger.info(f"Loading CSV: {file_path.name}")
        docs = CSVLoader(str(file_path), encoding="utf-8").load()
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
        """Process a file upload from a BinaryIO stream."""
        docs = self.load_from_upload(file, filename)
        return self.split_documents(docs)

    @track(name="document_processor_process")
    def process_upload_bytes(self, file_bytes: bytes, filename: str) -> list[Document]:
        """
        Process a file upload from raw bytes, avoiding redundant BytesIO wrapping.
        Writes to a single temp file for the underlying loader to read.
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
