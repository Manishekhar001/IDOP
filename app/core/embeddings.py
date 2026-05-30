from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    logger.info(
        "Initializing OpenAI embeddings: model=text-embedding-3-small standard embedding dimensions"
    )
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model="text-embedding-3-small", # standard dimensions is 1536
    )
    logger.info("OpenAI embeddings initialized successfully")
    return embeddings


class EmbeddingsService:
    def __init__(self) -> None:
        self.embeddings = get_embeddings()

    @track(name="embeddings_embed_query")
    def embed_query(self, text: str) -> list[float]:
        logger.debug(f"Embedding query: {text[:60]}...")
        return self.embeddings.embed_query(text)

    @track(name="embeddings_embed_documents")
    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        logger.debug(f"Embedding {len(docs)} documents")
        return self.embeddings.embed_documents(docs)
