from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            ".env",
            "backend/.env",
            ".env.local",
            "backend/.env.local",
            ".env.private",
            "backend/.env.private",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="LLM_MODEL")
    llm_supports_tools: Optional[bool] = Field(
        default=None, validation_alias="LLM_SUPPORTS_TOOLS"
    )
    llm_temperature: Optional[float] = Field(
        default=None, validation_alias="LLM_TEMPERATURE"
    )
    llm_top_p: Optional[float] = Field(default=None, validation_alias="LLM_TOP_P")
    llm_max_tokens: Optional[int] = Field(
        default=None, validation_alias="LLM_MAX_TOKENS"
    )
    llm_deep_think: Optional[bool] = Field(
        default=None, validation_alias="LLM_DEEP_THINK"
    )
    llm_reasoning_effort: Optional[str] = Field(
        default=None, validation_alias="LLM_REASONING_EFFORT"
    )
    llm_model_kwargs_json: Optional[dict[str, Any]] = Field(
        default=None, validation_alias="LLM_MODEL_KWARGS_JSON"
    )
    llm_extra_body_json: Optional[dict[str, Any]] = Field(
        default=None, validation_alias="LLM_EXTRA_BODY_JSON"
    )
    embedding_provider: str = Field(
        default="", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(default="", validation_alias="EMBEDDING_MODEL")

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

    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", validation_alias="OPENAI_BASE_URL")
    openai_model_supports_tools: bool = Field(
        default=True, validation_alias="OPENAI_MODEL_SUPPORTS_TOOLS"
    )
    openai_embed_model: str = Field(
        default="text-embedding-3-small", validation_alias="OPENAI_EMBED_MODEL"
    )
    openai_temperature: Optional[float] = Field(
        default=None, validation_alias="OPENAI_TEMPERATURE"
    )
    openai_top_p: Optional[float] = Field(
        default=None, validation_alias="OPENAI_TOP_P"
    )
    openai_max_tokens: Optional[int] = Field(
        default=None, validation_alias="OPENAI_MAX_TOKENS"
    )
    openai_deep_think: Optional[bool] = Field(
        default=None, validation_alias="OPENAI_DEEP_THINK"
    )
    openai_reasoning_effort: Optional[str] = Field(
        default=None, validation_alias="OPENAI_REASONING_EFFORT"
    )
    openai_model_kwargs_json: Optional[dict[str, Any]] = Field(
        default=None, validation_alias="OPENAI_MODEL_KWARGS_JSON"
    )
    openai_extra_body_json: Optional[dict[str, Any]] = Field(
        default=None, validation_alias="OPENAI_EXTRA_BODY_JSON"
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

    react_max_steps: int = Field(default=4, validation_alias="REACT_MAX_STEPS")
    react_tool_timeout_sec: float = Field(
        default=15.0, validation_alias="REACT_TOOL_TIMEOUT_SEC"
    )
    react_stream_thoughts: bool = Field(
        default=False, validation_alias="REACT_STREAM_THOUGHTS"
    )
    enable_explicit_react_runtime: bool = Field(
        default=False, validation_alias="ENABLE_EXPLICIT_REACT_RUNTIME"
    )

    job_queue_max: int = Field(default=256, validation_alias="JOB_QUEUE_MAX")
    async_summary_update: bool = Field(
        default=True, validation_alias="ASYNC_SUMMARY_UPDATE"
    )

    enable_postgres_persistence: bool = Field(
        default=False, validation_alias="ENABLE_POSTGRES_PERSISTENCE"
    )
    postgres_dsn: str = Field(default="", validation_alias="POSTGRES_DSN")
    postgres_connect_timeout: float = Field(
        default=3.0, validation_alias="POSTGRES_CONNECT_TIMEOUT"
    )
    postgres_restore_on_redis_miss: bool = Field(
        default=False,
        validation_alias="POSTGRES_RESTORE_ON_REDIS_MISS",
    )
    postgres_restore_turn_limit: int = Field(
        default=20,
        validation_alias="POSTGRES_RESTORE_TURN_LIMIT",
    )
    postgres_restore_redis_ttl_sec: int = Field(
        default=86400 * 7,
        validation_alias="POSTGRES_RESTORE_REDIS_TTL_SEC",
    )

    # 后台复盘学习层（Hermes Background Review 范式，默认全关，opt-in）
    enable_background_review: bool = Field(
        default=False, validation_alias="ENABLE_BACKGROUND_REVIEW"
    )
    review_memory_interval: int = Field(
        default=3, validation_alias="REVIEW_MEMORY_INTERVAL"
    )
    review_skills_interval: int = Field(
        default=5, validation_alias="REVIEW_SKILLS_INTERVAL"
    )
    review_base_dir: str = Field(
        default="agent_learning", validation_alias="REVIEW_BASE_DIR"
    )
    review_max_items: int = Field(default=5, validation_alias="REVIEW_MAX_ITEMS")
    review_model: str = Field(default="", validation_alias="REVIEW_MODEL")
    review_force_structured: bool = Field(
        default=False, validation_alias="REVIEW_FORCE_STRUCTURED"
    )
    review_tool_trace: bool = Field(
        default=False, validation_alias="REVIEW_TOOL_TRACE"
    )
    inject_learning_into_prompt: bool = Field(
        default=False, validation_alias="INJECT_LEARNING_INTO_PROMPT"
    )
    learning_prompt_budget_chars: int = Field(
        default=1200, validation_alias="LEARNING_PROMPT_BUDGET_CHARS"
    )

    # 工具调用安全加固层（Tool Guard，对标 OWASP LLM08 Excessive Agency）
    # 安全默认：enforce 启动即生效；随仓库策略放行现有只读工具、拒绝未知工具。
    tool_guard_mode: str = Field(
        default="enforce", validation_alias="TOOL_GUARD_MODE"
    )  # off | log | enforce
    tool_policy_file: str = Field(
        default="backend/config/tool_policy.yaml",
        validation_alias="TOOL_POLICY_FILE",
    )
    max_tool_calls_per_turn: int = Field(
        default=8, validation_alias="MAX_TOOL_CALLS_PER_TURN"
    )
    tool_scrub_output: bool = Field(
        default=True, validation_alias="TOOL_SCRUB_OUTPUT"
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", validation_alias="LOG_DIR")
    app_log_file: str = Field(default="backend.src.slothbearflow_backend.log", validation_alias="APP_LOG_FILE")
    access_log_file: str = Field(
        default="access.log", validation_alias="ACCESS_LOG_FILE"
    )
    error_log_file: str = Field(
        default="error.log", validation_alias="ERROR_LOG_FILE"
    )

    @field_validator("llm_supports_tools", mode="before")
    @classmethod
    def _empty_llm_supports_tools_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator(
        "llm_reasoning_effort",
        "openai_reasoning_effort",
        mode="before",
    )
    @classmethod
    def _empty_reasoning_effort_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator(
        "llm_model_kwargs_json",
        "llm_extra_body_json",
        "openai_model_kwargs_json",
        "openai_extra_body_json",
        mode="before",
    )
    @classmethod
    def _parse_json_dict_or_none(cls, value: object) -> object:
        if value in (None, ""):
            return None
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("JSON config must be an object")
        raise ValueError("JSON config must be a JSON object string")


@lru_cache
def get_settings() -> Settings:
    return Settings()
