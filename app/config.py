from functools import lru_cache
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    @model_validator(mode="after")
    def normalize_database_url(self) -> "Settings":
        if not getattr(self, "supabase_db_url", None):
            self.supabase_db_url = self.database_url or ""
            
        for attr in ["database_url", "supabase_db_url"]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val and val.startswith("postgresql+psycopg://"):
                    setattr(self, attr, val.replace("postgresql+psycopg://", "postgresql://", 1))
        return self




    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General App Config
    app_name: str = "Intelligent Data Operations Platform (IDOP)"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = "*"

    # OpenAI API Configuration
    openai_api_key: str
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    memory_llm_model: str = "gpt-4o-mini"
    memory_llm_temperature: float = 0.0

    # Qdrant Vector DB Configuration
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "idop_documents"
    embedding_dimension: int = 1536

    # Relational Database Configuration
    database_url: str
    supabase_db_url: str = ""

    # Storage Backend Configuration (for bulk document ingestion cache)
    storage_backend: str = "local"  # 'local' or 's3'
    s3_cache_bucket: str = "idop-cache-docs"
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Upstash Redis Configuration (for query/SQL cache)
    upstash_redis_url: Optional[str] = None
    upstash_redis_token: Optional[str] = None

    # Cache TTL Configurations (in seconds)
    cache_ttl_embeddings: int = 604800  # 7 days
    cache_ttl_rag: int = 3600           # 1 hour
    cache_ttl_sql_gen: int = 86400      # 24 hours
    cache_ttl_sql_result: int = 900     # 15 minutes

    # Search & Reranking APIs
    tavily_api_key: str
    tavily_max_results: int = 5
    voyage_api_key: Optional[str] = None

    # LangGraph State Configuration
    stm_message_threshold: int = 6
    crag_upper_threshold: float = 0.7
    crag_lower_threshold: float = 0.3
    srag_max_retries: int = 2
    max_rewrite_tries: int = 2

    # Chunking settings
    chunk_size: int = 512
    chunk_overlap: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
