from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Single source of truth for the application version.
# Imported by main.py and the settings object.
__version__ = "0.1.1"


class Settings(BaseSettings):
    @model_validator(mode="after")
    def normalize_database_url(self) -> "Settings":
        import urllib.parse

        if not getattr(self, "supabase_db_url", None):
            self.supabase_db_url = self.database_url or ""

        for attr in ["database_url", "supabase_db_url"]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val:
                    if val.startswith("postgresql+psycopg://"):
                        val = val.replace("postgresql+psycopg://", "postgresql://", 1)

                    try:
                        parsed = urllib.parse.urlsplit(val)
                        if parsed.password:
                            encoded_pass = urllib.parse.quote(parsed.password, safe="")
                            if parsed.port:
                                netloc = f"{parsed.username}:{encoded_pass}@{parsed.hostname}:{parsed.port}"
                            else:
                                netloc = f"{parsed.username}:{encoded_pass}@{parsed.hostname}"

                            val = urllib.parse.urlunsplit(
                                (
                                    parsed.scheme,
                                    netloc,
                                    parsed.path,
                                    parsed.query,
                                    parsed.fragment,
                                )
                            )
                        setattr(self, attr, val)
                    except Exception:
                        pass
        return self

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # General App Config
    app_name: str = "Intelligent Data Operations Platform (IDOP)"
    app_version: str = __version__
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = "*"
    # JWT Authentication
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours
    # Rate Limiting
    rate_limit: str = "60/minute"
    git_commit_sha: str = "unknown"

    # LLM Provider Configuration
    openai_api_key: str
    groq_api_key: str | None = None
    groq_api_key_1: str | None = None
    groq_api_key_2: str | None = None
    groq_api_key_3: str | None = None
    groq_api_key_4: str | None = None

    llm_provider: str = "litellm"  # "litellm" or "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    memory_llm_model: str = "llama-3.3-70b-versatile"
    memory_llm_temperature: float = 0.0
    vanna_llm_model: str = "gpt-4o-mini"

    # Embedding Provider Configuration
    embedding_provider: str = "nomic"  # "nomic" or "voyage"
    voyage_api_key: str | None = (
        None  # Voyage API key (used for both embeddings and reranking)
    )
    voyage_embedding_model: str = "voyage-3"
    voyage_embedding_dimension: int = 1024
    nomic_api_key: str | None = None
    nomic_embedding_model: str = "nomic-embed-text-v1.5"
    nomic_embedding_dimension: int = 768

    @property
    def embedding_dimension(self) -> int:
        """Return the correct embedding dimension based on the active provider."""
        if self.embedding_provider == "voyage":
            return self.voyage_embedding_dimension
        elif self.embedding_provider == "nomic":
            return self.nomic_embedding_dimension
        raise ValueError(
            f"Unsupported embedding provider '{self.embedding_provider}'. Supported: voyage, nomic"
        )

    @property
    def groq_api_keys(self) -> list[str]:
        """Return all configured non-empty Groq API keys."""
        keys = []
        for attr in [
            "groq_api_key_1",
            "groq_api_key_2",
            "groq_api_key_3",
            "groq_api_key_4",
        ]:
            val = getattr(self, attr, None)
            if val and str(val).strip():
                keys.append(str(val).strip())
        if (
            self.groq_api_key
            and str(self.groq_api_key).strip()
            and str(self.groq_api_key).strip() not in keys
        ):
            keys.append(str(self.groq_api_key).strip())
        return keys

    # Qdrant Vector DB Configuration
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "idop_documents"

    # Relational Database Configuration
    database_url: str
    supabase_db_url: str = ""

    # Storage Backend Configuration (for bulk document ingestion cache)
    storage_backend: str = (
        "s3"  # 's3' or 'local' — override via STORAGE_BACKEND in .env
    )
    s3_cache_bucket: str = "idop-cache-docs"
    cache_dir: str = "data/cached_chunks"  # local storage directory for document chunks
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    # Upstash Redis Configuration (for query/SQL cache)
    upstash_redis_url: str | None = None
    upstash_redis_token: str | None = None

    # Cache TTL Configurations (in seconds)
    cache_ttl_embeddings: int = 604800  # 7 days
    cache_ttl_rag: int = 3600  # 1 hour
    cache_ttl_sql_gen: int = 86400  # 24 hours
    cache_ttl_sql_result: int = 900  # 15 minutes

    # Search & Reranking APIs
    tavily_api_key: str
    tavily_max_results: int = 5
    # Voyage reranking uses the same voyage_api_key as embeddings above.
    # voyage_rerank_api_key is deprecated - use voyage_api_key instead.

    # LangGraph State Configuration
    stm_message_threshold: int = 6
    crag_upper_threshold: float = 0.7
    crag_lower_threshold: float = 0.3
    srag_max_retries: int = 2
    max_rewrite_tries: int = 2
    retrieval_k: int = 5

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    # OPIK Observability Configuration (optional)
    opik_api_key: str | None = None
    opik_workspace: str | None = None
    opik_project_name: str | None = None

    # JWT / Authentication Configuration
    jwt_secret_key: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60


@lru_cache
def get_settings() -> Settings:
    return Settings()
