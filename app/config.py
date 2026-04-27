from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_model: str = Field(default="deepseek-r1:7b", validation_alias="OLLAMA_MODEL")
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434", validation_alias="OLLAMA_BASE_URL"
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text", validation_alias="OLLAMA_EMBED_MODEL"
    )
    ollama_model_supports_tools: bool = Field(
        default=False, validation_alias="OLLAMA_MODEL_SUPPORTS_TOOLS"
    )

    redis_host: str = Field(default="127.0.0.1", validation_alias="REDIS_HOST")
    redis_port: int = Field(default=6379, validation_alias="REDIS_PORT")
    redis_db: int = Field(default=0, validation_alias="REDIS_DB")
    redis_password: Optional[str] = Field(
        default=None, validation_alias="REDIS_PASSWORD"
    )
    redis_socket_connect_timeout: float = Field(
        default=0.5, validation_alias="REDIS_SOCKET_CONNECT_TIMEOUT"
    )
    redis_socket_timeout: float = Field(
        default=0.5, validation_alias="REDIS_SOCKET_TIMEOUT"
    )

    milvus_uri: str = Field(
        default="http://127.0.0.1:19530", validation_alias="MILVUS_URI"
    )
    milvus_collection: str = Field(
        default="chat_knowledge", validation_alias="MILVUS_COLLECTION"
    )
    milvus_timeout: float = Field(default=1.0, validation_alias="MILVUS_TIMEOUT")
    skip_milvus: bool = Field(default=False, validation_alias="SKIP_MILVUS")
    use_rag: bool = Field(default=True, validation_alias="USE_RAG")

    memory_window_pairs: int = Field(default=6, validation_alias="MEMORY_WINDOW_PAIRS")
    stream_output: bool = Field(default=False, validation_alias="STREAM_OUTPUT")
    stream_output_format: str = Field(
        default="plain", validation_alias="STREAM_OUTPUT_FORMAT"
    )
    structured_output: bool = Field(
        default=False, validation_alias="STRUCTURED_OUTPUT"
    )

    job_queue_max: int = Field(default=256, validation_alias="JOB_QUEUE_MAX")
    async_summary_update: bool = Field(
        default=True, validation_alias="ASYNC_SUMMARY_UPDATE"
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", validation_alias="LOG_DIR")
    app_log_file: str = Field(default="app.log", validation_alias="APP_LOG_FILE")
    access_log_file: str = Field(
        default="access.log", validation_alias="ACCESS_LOG_FILE"
    )
    error_log_file: str = Field(
        default="error.log", validation_alias="ERROR_LOG_FILE"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
