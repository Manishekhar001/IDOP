import time
from functools import lru_cache, wraps

from langchain_openai import OpenAIEmbeddings
from app.config import get_settings
from app.opik import track
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIQuotaError(Exception):
    """Raised when OpenAI returns HTTP 429 (insufficient quota / rate limit)."""
    pass


def _retry_on_quota(
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff: float = 4.0,
):
    """
    Decorator that retries an OpenAI API call with exponential backoff
    on HTTP 429 (insufficient_quota / rate_limit) errors.

    After exhausting all retries, raises OpenAIQuotaError with a clear
    message pointing the user to the OpenAI billing dashboard.
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
                    )
                    if not is_quota:
                        raise  # Non-quota errors re-raise immediately

                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (backoff ** (attempt - 1))
                        logger.warning(
                            f"OpenAI quota/rate-limit hit (attempt {attempt}/{max_retries}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"OpenAI quota exhausted after {max_retries} retries. "
                            f"The API key needs more credits."
                        )

            raise OpenAIQuotaError(
                "OpenAI API quota exhausted. The deployed API key has insufficient credits.\n"
                "To resolve this:\n"
                "1. Visit https://platform.openai.com/account/billing to add credits\n"
                "2. Or set a new OPENAI_API_KEY with available quota in the deployment secrets\n"
                f"Original error: {last_exc}"
            ) from last_exc
        return wrapper
    return decorator


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    logger.info(
        "Initializing OpenAI embeddings: model=text-embedding-3-small standard embedding dimensions"
    )
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model="text-embedding-3-small",  # standard dimensions is 1536
    )
    logger.info("OpenAI embeddings initialized successfully")
    return embeddings


class EmbeddingsService:
    def __init__(self) -> None:
        self.embeddings = get_embeddings()
        # Wrap the raw embed methods with retry logic
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
