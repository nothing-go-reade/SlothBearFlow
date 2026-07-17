from __future__ import annotations

import ipaddress
import json
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _valid_password_hash(value: object) -> bool:
    try:
        algorithm, raw_iterations, salt, digest = str(value or "").split("$", 3)
        iterations = int(raw_iterations)
    except (TypeError, ValueError):
        return False
    encoded_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return bool(
        algorithm == "pbkdf2_sha256"
        and 100_000 <= iterations <= 2_000_000
        and len(salt) >= 16
        and len(digest) >= 32
        and set(salt) <= encoded_chars
        and set(digest) <= encoded_chars
    )


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

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    app_release: str = Field(default="dev", validation_alias="APP_RELEASE")

    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="LLM_MODEL")
    llm_supports_tools: Optional[bool] = Field(
        default=None, validation_alias="LLM_SUPPORTS_TOOLS"
    )
    llm_healthcheck_enabled: bool = Field(
        default=False, validation_alias="LLM_HEALTHCHECK_ENABLED"
    )
    llm_healthcheck_timeout_sec: float = Field(
        default=2.0, validation_alias="LLM_HEALTHCHECK_TIMEOUT_SEC"
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
    redis_retry_interval_sec: float = Field(
        default=5.0, validation_alias="REDIS_RETRY_INTERVAL_SEC"
    )

    milvus_uri: str = Field(
        default="http://127.0.0.1:19530", validation_alias="MILVUS_URI"
    )
    milvus_collection: str = Field(
        default="chat_knowledge", validation_alias="MILVUS_COLLECTION"
    )
    milvus_timeout: float = Field(default=1.0, validation_alias="MILVUS_TIMEOUT")
    milvus_retry_interval_sec: float = Field(
        default=10.0, validation_alias="MILVUS_RETRY_INTERVAL_SEC"
    )
    skip_milvus: bool = Field(default=False, validation_alias="SKIP_MILVUS")
    milvus_token: str = Field(default="", validation_alias="MILVUS_TOKEN")
    use_rag: bool = Field(default=True, validation_alias="USE_RAG")
    rag_chunk_size: int = Field(default=600, validation_alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=100, validation_alias="RAG_CHUNK_OVERLAP")
    rag_retrieval_top_k: int = Field(
        default=24, validation_alias="RAG_RETRIEVAL_TOP_K"
    )
    rag_max_context_chunks: int = Field(
        default=5, validation_alias="RAG_MAX_CONTEXT_CHUNKS"
    )
    rag_context_max_tokens: int = Field(
        default=4000, validation_alias="RAG_CONTEXT_MAX_TOKENS"
    )
    rag_multi_query: bool = Field(default=True, validation_alias="RAG_MULTI_QUERY")
    rag_rrf_k: int = Field(default=60, validation_alias="RAG_RRF_K")
    rag_relevance_threshold: float = Field(
        default=0.12, validation_alias="RAG_RELEVANCE_THRESHOLD"
    )
    rag_reranker_provider: str = Field(
        default="lexical", validation_alias="RAG_RERANKER_PROVIDER"
    )
    rag_cross_encoder_model: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias="RAG_CROSS_ENCODER_MODEL",
    )
    rag_block_prompt_injection: bool = Field(
        default=True, validation_alias="RAG_BLOCK_PROMPT_INJECTION"
    )
    rag_allow_legacy_documents: bool = Field(
        default=True, validation_alias="RAG_ALLOW_LEGACY_DOCUMENTS"
    )

    memory_window_pairs: int = Field(default=6, validation_alias="MEMORY_WINDOW_PAIRS")
    memory_window_max_tokens: int = Field(
        default=6000, validation_alias="MEMORY_WINDOW_MAX_TOKENS"
    )
    memory_ttl_sec: int = Field(default=604800, validation_alias="MEMORY_TTL_SEC")
    memory_max_messages: int = Field(
        default=200, validation_alias="MEMORY_MAX_MESSAGES"
    )
    memory_redact_pii: bool = Field(
        default=True, validation_alias="MEMORY_REDACT_PII"
    )
    stream_output: bool = Field(default=False, validation_alias="STREAM_OUTPUT")
    stream_output_format: str = Field(
        default="plain", validation_alias="STREAM_OUTPUT_FORMAT"
    )
    structured_output: bool = Field(
        default=False, validation_alias="STRUCTURED_OUTPUT"
    )

    react_max_steps: int = Field(default=4, validation_alias="REACT_MAX_STEPS")
    enable_explicit_react_runtime: bool = Field(
        default=False, validation_alias="ENABLE_EXPLICIT_REACT_RUNTIME"
    )
    agent_timeout_sec: float = Field(
        default=90.0, validation_alias="AGENT_TIMEOUT_SEC"
    )
    agent_max_iterations: int = Field(
        default=4, validation_alias="AGENT_MAX_ITERATIONS"
    )
    agent_prompt_version: str = Field(
        default="v1", validation_alias="AGENT_PROMPT_VERSION"
    )
    job_queue_max: int = Field(default=256, validation_alias="JOB_QUEUE_MAX")
    ingest_max_attempts: int = Field(default=3, validation_alias="INGEST_MAX_ATTEMPTS")
    ingest_retry_backoff_sec: float = Field(
        default=5.0, validation_alias="INGEST_RETRY_BACKOFF_SEC"
    )
    async_summary_update: bool = Field(
        default=True, validation_alias="ASYNC_SUMMARY_UPDATE"
    )
    summary_retry_attempts: int = Field(
        default=2, validation_alias="SUMMARY_RETRY_ATTEMPTS"
    )
    summary_timeout_sec: float = Field(
        default=60.0, validation_alias="SUMMARY_TIMEOUT_SEC"
    )
    summary_input_max_chars: int = Field(
        default=24000, validation_alias="SUMMARY_INPUT_MAX_CHARS"
    )

    enable_postgres_persistence: bool = Field(
        default=False, validation_alias="ENABLE_POSTGRES_PERSISTENCE"
    )
    postgres_dsn: str = Field(default="", validation_alias="POSTGRES_DSN")
    postgres_connect_timeout: float = Field(
        default=3.0, validation_alias="POSTGRES_CONNECT_TIMEOUT"
    )
    postgres_pool_min_size: int = Field(
        default=1, validation_alias="POSTGRES_POOL_MIN_SIZE"
    )
    postgres_pool_max_size: int = Field(
        default=10, validation_alias="POSTGRES_POOL_MAX_SIZE"
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
    review_min_confidence: float = Field(
        default=0.75, validation_alias="REVIEW_MIN_CONFIDENCE"
    )
    review_model: str = Field(default="", validation_alias="REVIEW_MODEL")
    review_force_structured: bool = Field(
        default=False, validation_alias="REVIEW_FORCE_STRUCTURED"
    )
    review_tool_trace: bool = Field(
        default=False, validation_alias="REVIEW_TOOL_TRACE"
    )
    review_timeout_sec: float = Field(
        default=90.0, validation_alias="REVIEW_TIMEOUT_SEC"
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
    tool_timeout_sec: float = Field(
        default=15.0, validation_alias="TOOL_TIMEOUT_SEC"
    )
    tool_retry_attempts: int = Field(
        default=1, validation_alias="TOOL_RETRY_ATTEMPTS"
    )
    tool_circuit_failure_threshold: int = Field(
        default=3, validation_alias="TOOL_CIRCUIT_FAILURE_THRESHOLD"
    )
    tool_circuit_recovery_sec: float = Field(
        default=30.0, validation_alias="TOOL_CIRCUIT_RECOVERY_SEC"
    )
    tool_trace_observation_max_chars: int = Field(
        default=800, validation_alias="TOOL_TRACE_OBSERVATION_MAX_CHARS"
    )
    tool_observation_max_chars: int = Field(
        default=12000, validation_alias="TOOL_OBSERVATION_MAX_CHARS"
    )
    tool_scrub_output: bool = Field(
        default=True, validation_alias="TOOL_SCRUB_OUTPUT"
    )

    mcp_enabled: bool = Field(default=False, validation_alias="MCP_ENABLED")
    mcp_servers_json: list[dict[str, Any]] = Field(
        default_factory=list, validation_alias="MCP_SERVERS_JSON"
    )
    mcp_tool_allowlist_json: list[str] = Field(
        default_factory=list, validation_alias="MCP_TOOL_ALLOWLIST_JSON"
    )
    mcp_allowed_hosts_json: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost"],
        validation_alias="MCP_ALLOWED_HOSTS_JSON",
    )
    mcp_timeout_sec: float = Field(default=10.0, validation_alias="MCP_TIMEOUT_SEC")
    mcp_discovery_ttl_sec: float = Field(
        default=60.0, validation_alias="MCP_DISCOVERY_TTL_SEC"
    )
    mcp_egress_proxy_url: str = Field(
        default="", validation_alias="MCP_EGRESS_PROXY_URL"
    )

    observability_enabled: bool = Field(
        default=True, validation_alias="OBSERVABILITY_ENABLED"
    )
    prometheus_enabled: bool = Field(
        default=True, validation_alias="PROMETHEUS_ENABLED"
    )
    metrics_bearer_token: str = Field(
        default="", validation_alias="METRICS_BEARER_TOKEN"
    )
    trace_store_size: int = Field(default=200, validation_alias="TRACE_STORE_SIZE")
    trace_include_content: bool = Field(
        default=False, validation_alias="TRACE_INCLUDE_CONTENT"
    )
    langfuse_enabled: bool = Field(
        default=False, validation_alias="LANGFUSE_ENABLED"
    )
    langfuse_host: str = Field(
        default="http://127.0.0.1:3000", validation_alias="LANGFUSE_HOST"
    )
    langfuse_public_key: str = Field(
        default="", validation_alias="LANGFUSE_PUBLIC_KEY"
    )
    langfuse_secret_key: str = Field(
        default="", validation_alias="LANGFUSE_SECRET_KEY"
    )

    auth_required: bool = Field(default=True, validation_alias="AUTH_REQUIRED")
    auth_secret: str = Field(default="", validation_alias="AUTH_SECRET")
    auth_issuer: str = Field(default="slothbearflow", validation_alias="AUTH_ISSUER")
    auth_token_ttl_sec: int = Field(
        default=3600, validation_alias="AUTH_TOKEN_TTL_SEC"
    )
    auth_users_json: dict[str, Any] = Field(
        default_factory=dict, validation_alias="AUTH_USERS_JSON"
    )
    auth_local_user_id: str = Field(
        default="local-user", validation_alias="AUTH_LOCAL_USER_ID"
    )
    auth_local_tenant_id: str = Field(
        default="local", validation_alias="AUTH_LOCAL_TENANT_ID"
    )
    auth_local_roles_json: list[str] = Field(
        default_factory=lambda: ["viewer"],
        validation_alias="AUTH_LOCAL_ROLES_JSON",
    )
    allow_insecure_local_network: bool = Field(
        default=False, validation_alias="ALLOW_INSECURE_LOCAL_NETWORK"
    )
    rate_limit_per_minute: int = Field(
        default=60, validation_alias="RATE_LIMIT_PER_MINUTE"
    )
    chat_concurrency_limit: int = Field(
        default=8, validation_alias="CHAT_CONCURRENCY_LIMIT"
    )
    concurrency_wait_sec: float = Field(
        default=2.0, validation_alias="CONCURRENCY_WAIT_SEC"
    )
    api_max_request_bytes: int = Field(
        default=2_000_000, validation_alias="API_MAX_REQUEST_BYTES"
    )
    cors_origins_json: list[str] = Field(
        default_factory=lambda: [
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        validation_alias="CORS_ORIGINS_JSON",
    )
    docs_enabled: bool = Field(default=True, validation_alias="DOCS_ENABLED")
    chat_message_max_chars: int = Field(
        default=20_000, validation_alias="CHAT_MESSAGE_MAX_CHARS"
    )
    ingest_text_max_chars: int = Field(
        default=1_000_000, validation_alias="INGEST_TEXT_MAX_CHARS"
    )
    tool_approval_ttl_sec: int = Field(
        default=900, validation_alias="TOOL_APPROVAL_TTL_SEC"
    )
    audit_enabled: bool = Field(default=True, validation_alias="AUDIT_ENABLED")
    audit_log_file: str = Field(
        default="logs/audit.jsonl", validation_alias="AUDIT_LOG_FILE"
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

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalize_app_env(cls, value: object) -> str:
        normalized = str(value or "local").strip().lower()
        if normalized not in {"local", "test", "staging", "production"}:
            raise ValueError("APP_ENV must be local, test, staging, or production")
        return normalized

    @field_validator("tool_guard_mode", mode="before")
    @classmethod
    def _validate_tool_guard_mode(cls, value: object) -> str:
        normalized = str(value or "enforce").strip().lower()
        if normalized not in {"off", "log", "enforce"}:
            raise ValueError("TOOL_GUARD_MODE must be off, log, or enforce")
        return normalized

    @field_validator("stream_output_format", mode="before")
    @classmethod
    def _validate_stream_format(cls, value: object) -> str:
        normalized = str(value or "plain").strip().lower()
        if normalized not in {"plain", "sse"}:
            raise ValueError("STREAM_OUTPUT_FORMAT must be plain or sse")
        return normalized

    @field_validator("rag_reranker_provider", mode="before")
    @classmethod
    def _validate_reranker_provider(cls, value: object) -> str:
        normalized = str(value or "lexical").strip().lower()
        if normalized not in {"none", "lexical", "cross_encoder"}:
            raise ValueError(
                "RAG_RERANKER_PROVIDER must be none, lexical, or cross_encoder"
            )
        return normalized

    @field_validator(
        "agent_timeout_sec",
        "tool_timeout_sec",
        "tool_circuit_recovery_sec",
        "redis_retry_interval_sec",
        "milvus_retry_interval_sec",
        "ingest_retry_backoff_sec",
        "mcp_timeout_sec",
        "mcp_discovery_ttl_sec",
        "concurrency_wait_sec",
        "summary_timeout_sec",
        "review_timeout_sec",
        "llm_healthcheck_timeout_sec",
    )
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if float(value) <= 0:
            raise ValueError("timeout values must be greater than zero")
        return float(value)

    @field_validator(
        "agent_max_iterations",
        "react_max_steps",
        "max_tool_calls_per_turn",
        "tool_circuit_failure_threshold",
        "tool_trace_observation_max_chars",
        "tool_observation_max_chars",
        "rag_chunk_size",
        "rag_retrieval_top_k",
        "rag_max_context_chunks",
        "rag_context_max_tokens",
        "rag_rrf_k",
        "memory_window_max_tokens",
        "memory_ttl_sec",
        "memory_max_messages",
        "summary_input_max_chars",
        "ingest_max_attempts",
        "trace_store_size",
        "auth_token_ttl_sec",
        "rate_limit_per_minute",
        "chat_concurrency_limit",
        "api_max_request_bytes",
        "chat_message_max_chars",
        "ingest_text_max_chars",
        "tool_approval_ttl_sec",
        "postgres_pool_min_size",
        "postgres_pool_max_size",
    )
    @classmethod
    def _positive_integer(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("limit values must be greater than zero")
        return int(value)

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _valid_chunk_overlap(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("RAG_CHUNK_OVERLAP cannot be negative")
        return int(value)

    @field_validator("rag_relevance_threshold")
    @classmethod
    def _valid_relevance_threshold(cls, value: float) -> float:
        if not 0 <= float(value) <= 1:
            raise ValueError("RAG_RELEVANCE_THRESHOLD must be between 0 and 1")
        return float(value)

    @field_validator("review_min_confidence")
    @classmethod
    def _valid_review_confidence(cls, value: float) -> float:
        if not 0 <= float(value) <= 1:
            raise ValueError("REVIEW_MIN_CONFIDENCE must be between 0 and 1")
        return float(value)

    @field_validator("tool_retry_attempts")
    @classmethod
    def _non_negative_retries(cls, value: int) -> int:
        if int(value) < 0 or int(value) > 5:
            raise ValueError("TOOL_RETRY_ATTEMPTS must be between 0 and 5")
        return int(value)

    @field_validator("summary_retry_attempts")
    @classmethod
    def _valid_summary_retries(cls, value: int) -> int:
        if int(value) < 0 or int(value) > 5:
            raise ValueError("SUMMARY_RETRY_ATTEMPTS must be between 0 and 5")
        return int(value)

    @field_validator("mcp_egress_proxy_url")
    @classmethod
    def _valid_mcp_egress_proxy(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            return ""
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("MCP_EGRESS_PROXY_URL must be an absolute http(s) URL")
        if parsed.username or parsed.password:
            raise ValueError("MCP_EGRESS_PROXY_URL cannot contain credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("MCP_EGRESS_PROXY_URL must identify a proxy origin")
        try:
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("MCP_EGRESS_PROXY_URL contains an invalid port") from exc
        return normalized

    @field_validator("allow_insecure_local_network")
    @classmethod
    def _reject_anonymous_network_bypass(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "ALLOW_INSECURE_LOCAL_NETWORK is no longer supported; "
                "anonymous mode is loopback-only"
            )
        return False

    @model_validator(mode="after")
    def _production_security_baseline(self) -> "Settings":
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.postgres_pool_min_size > self.postgres_pool_max_size:
            raise ValueError(
                "POSTGRES_POOL_MIN_SIZE cannot exceed POSTGRES_POOL_MAX_SIZE"
            )
        if self.app_env == "production" and self.tool_guard_mode != "enforce":
            raise ValueError("production requires TOOL_GUARD_MODE=enforce")
        if self.app_env in {"staging", "production"} and not self.auth_required:
            raise ValueError(f"{self.app_env} requires AUTH_REQUIRED=true")
        if self.app_env == "production":
            if len(self.auth_secret) < 32:
                raise ValueError(
                    "production AUTH_SECRET must contain at least 32 characters"
                )
            if len(str(self.redis_password or "")) < 12:
                raise ValueError(
                    "production REDIS_PASSWORD must contain at least 12 characters"
                )
            active_auth_users = 0
            auth_identities = set()
            for username, raw_user in self.auth_users_json.items():
                if not str(username).strip() or not isinstance(raw_user, dict):
                    raise ValueError("production auth users must be named objects")
                if raw_user.get("disabled"):
                    continue
                if not _valid_password_hash(raw_user.get("password_hash")):
                    raise ValueError(
                        f"production auth user {username!r} requires a valid PBKDF2 hash"
                    )
                tenant_id = str(raw_user.get("tenant_id") or "default").strip()
                user_id = str(raw_user.get("user_id") or username).strip()
                if (
                    not tenant_id
                    or not user_id
                    or len(tenant_id) > 128
                    or len(user_id) > 128
                    or any(
                        ord(character) < 32 or ord(character) == 127
                        for character in tenant_id + user_id
                    )
                ):
                    raise ValueError(
                        f"production auth user {username!r} has an invalid identity"
                    )
                identity = (tenant_id, user_id)
                if identity in auth_identities:
                    raise ValueError(
                        "production auth users must have unique "
                        "(tenant_id, user_id) identities"
                    )
                auth_identities.add(identity)
                active_auth_users += 1
            if not active_auth_users:
                raise ValueError("production requires at least one configured auth user")
            if self.rag_allow_legacy_documents:
                raise ValueError(
                    "production requires RAG_ALLOW_LEGACY_DOCUMENTS=false"
                )
            if not self.memory_redact_pii:
                raise ValueError("production requires MEMORY_REDACT_PII=true")
            if not self.llm_healthcheck_enabled:
                raise ValueError("production requires LLM_HEALTHCHECK_ENABLED=true")
            if self.use_rag:
                milvus_user, separator, milvus_password = str(
                    self.milvus_token
                ).strip().partition(":")
                if (
                    not separator
                    or not milvus_user
                    or len(milvus_password) < 12
                    or self.milvus_token == "root:Milvus"
                ):
                    raise ValueError(
                        "production RAG requires a strong username:password MILVUS_TOKEN"
                    )
            if self.use_rag and (
                not self.enable_postgres_persistence
                or not str(self.postgres_dsn).strip()
            ):
                raise ValueError(
                    "production RAG requires PostgreSQL persistence for durable ingestion"
                )
            if self.enable_postgres_persistence:
                try:
                    postgres_url = urlparse(str(self.postgres_dsn).strip())
                    _ = postgres_url.port
                except ValueError as exc:
                    raise ValueError("production POSTGRES_DSN is invalid") from exc
                if (
                    postgres_url.scheme not in {"postgres", "postgresql"}
                    or not postgres_url.hostname
                    or not postgres_url.username
                    or postgres_url.password in {None, ""}
                    or not postgres_url.path.strip("/")
                ):
                    raise ValueError(
                        "production POSTGRES_DSN must be a complete PostgreSQL URL"
                    )
                if unquote(postgres_url.username).lower() in {"postgres", "root"}:
                    raise ValueError(
                        "production POSTGRES_DSN must use a CRUD-only runtime role"
                    )
            if "*" in self.cors_origins_json:
                raise ValueError("production CORS origins cannot contain '*'")
            if self.prometheus_enabled and len(self.metrics_bearer_token) < 24:
                raise ValueError(
                    "production Prometheus endpoint requires METRICS_BEARER_TOKEN"
                )
            if self.mcp_enabled:
                if not self.mcp_servers_json or not self.mcp_tool_allowlist_json:
                    raise ValueError(
                        "production MCP requires servers and an explicit tool allowlist"
                    )
                for server in self.mcp_servers_json:
                    if not isinstance(server, dict):
                        raise ValueError("production MCP server entries must be objects")
                    if not server.get("allowed_tenants"):
                        raise ValueError(
                            "production MCP servers require allowed_tenants"
                        )
                    if not server.get("allowed_scopes"):
                        raise ValueError(
                            "production MCP servers require allowed_scopes"
                        )
                    if (
                        not _mcp_url_is_literal_loopback(server.get("url"))
                        and not self.mcp_egress_proxy_url
                    ):
                        raise ValueError(
                            "production external MCP servers require "
                            "MCP_EGRESS_PROXY_URL"
                        )
        return self

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

    @field_validator(
        "mcp_servers_json",
        "mcp_tool_allowlist_json",
        "mcp_allowed_hosts_json",
        "auth_local_roles_json",
        "cors_origins_json",
        mode="before",
    )
    @classmethod
    def _parse_json_list(cls, value: object) -> object:
        if value in (None, ""):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        raise ValueError("list settings must be JSON arrays")

    @field_validator("auth_users_json", mode="before")
    @classmethod
    def _parse_auth_users(cls, value: object) -> object:
        if value in (None, ""):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("AUTH_USERS_JSON must be a JSON object")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _mcp_url_is_literal_loopback(value: object) -> bool:
    try:
        raw_hostname = urlparse(str(value or "")).hostname
    except ValueError:
        return False
    hostname = str(raw_hostname or "").strip().lower().rstrip(".")
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
