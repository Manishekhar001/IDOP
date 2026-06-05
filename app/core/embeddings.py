"""
Embedding Service — supports OpenAI (text-embedding-3-small) and Voyage AI (voyage-3).

Configuration (in .env):
    EMBEDDING_PROVIDER=openai   # "openai" or "voyage"
    OPENAI_API_KEY=sk-...       # Required if EMBEDDING_PROVIDER=openai
    # OR
    VOYAGE_API_KEY=pa-...       # Required if EMBEDDING_PROVIDER=voyage
    VOYAGE_EMBEDDING_MODEL=voyage-3  # Optional, defaults to voyage-3
"""

import time
from functools import lru_cache, wraps
from typing import Any

from app.config import get_settings
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingQuotaError(Exception):
    """Raised when the embedding provider returns HTTP 429 (insufficient quota)."""

    pass


def _retry_on_quota(
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff: float = 4.0,
):
    """
    Decorator that retries an embedding API call with exponential backoff
    on HTTP 429 (rate limit / quota) errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    err_str = str(e).lower()
                    is_quota = (
                        "429" in err_str
                        or "insufficient_quota" in err_str
                        or "rate limit" in err_str
                        or "quota exceeded" in err_str
                        or "insufficient_quota" in err_str
                    )
                    if not is_quota:
                        raise  # Non-quota errors re-raise immediately

                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff ** (attempt - 1))
                        logger.warning(
                            f"Embedding quota/rate-limit hit (attempt {attempt}/{max_retries}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Embedding quota exhausted after {max_retries} retries."
                        )

            raise EmbeddingQuotaError(
                f"Embedding API quota exhausted after {max_retries} retries.\n"
                f"Original error: {last_exc}"
            ) from last_exc

        return wrapper

    return decorator


def _create_embedding_model() -> Any:
    """
    Create and return an embedding model based on settings.
    Returns an object with .embed_query(text) and .embed_documents(texts) methods.
    """
    settings = get_settings()
    provider = settings.embedding_provider

    if provider == "voyage":
        voyage_api_key = settings.voyage_api_key
        if not voyage_api_key:
            raise ValueError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY is not set.")
        try:
            from langchain_voyageai import VoyageAIEmbeddings

            model = settings.voyage_embedding_model or "voyage-3"
            logger.info(f"Initializing Voyage embeddings: model={model}")
            return VoyageAIEmbeddings(
                voyage_api_key=voyage_api_key,
                model=model,
            )
        except ImportError:
            logger.warning(
                "langchain-voyageai is not installed. "
                "Run: uv pip install langchain-voyageai"
            )
            raise
    else:
        # Default: OpenAI
        from langchain_openai import OpenAIEmbeddings

        logger.info("Initializing OpenAI embeddings: model=text-embedding-3-small")
        return OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-3-small",
        )


@lru_cache
def get_embeddings() -> Any:
    """Get or create the embedding model (cached)."""
    return _create_embedding_model()


class EmbeddingsService:
    """Unified embedding service supporting both OpenAI and Voyage."""

    def __init__(self) -> None:
        self.embeddings = get_embeddings()
        # Wrap with retry logic
        self.embed_query = _retry_on_quota()(self._embed_query)
        self.embed_documents = _retry_on_quota()(self._embed_documents)

    @track(name="embeddings_embed_query")
    def _embed_query(self, text: str) -> list[float]:
        logger.debug(f"Embedding query: {text[:60]}...")
        return self.embeddings.embed_query(text)

    @track(name="embeddings_embed_documents")
    def _embed_documents(self, docs: list[str]) -> list[list[float]]:
        logger.debug(f"Embedding {len(docs)} documents")
        return self.embeddings.embed_documents(docs)
