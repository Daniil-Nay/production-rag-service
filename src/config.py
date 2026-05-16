"""Application configuration backed by pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_base_url: str
    llm_api_key: str
    llm_model_primary: str
    llm_model_cheap: str
    llm_temperature: float = 0.1
    llm_max_tokens: int = 500
    request_timeout_s: int = 30
    max_retries: int = 3

    # Embeddings
    embed_model: str = "intfloat/multilingual-e5-base"
    embed_dimension: int = 768

    # Postgres / pgvector
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "rag"
    pg_password: str = "ragpass"
    pg_database: str = "ragdb"

    # Retrieval params
    chunk_size_words: int = 200
    chunk_overlap_words: int = 40
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5

    # Rate limiting
    rate_limit_per_minute: int = 60

    # Logging
    log_level: str = "INFO"

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


settings = Settings()  # type: ignore[call-arg]
