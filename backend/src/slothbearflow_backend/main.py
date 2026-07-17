from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from uvicorn.logging import AccessFormatter

from backend.src.slothbearflow_backend import build_agent_executor, get_settings
from backend.src.slothbearflow_backend.agent.conversation_loop import (
    AgentPreparationTimeout,
    ChatTurnRunner,
    TurnInput,
)
from backend.src.slothbearflow_backend.llm import (
    get_llm_status,
    llm_supports_tools,
)
from backend.src.slothbearflow_backend.learning.store import (
    LearningStore,
    learning_dir_for,
)
from backend.src.slothbearflow_backend.rag.embedding import get_embedding_model_name, get_embedding_provider
from backend.src.slothbearflow_backend.deps import InMemoryRedis, ping_redis
from backend.src.slothbearflow_backend.memory.redis_memory import get_redis_session
from backend.src.slothbearflow_backend.memory.redis_memory import (
    delete_session_payload,
    get_redis_session_generation,
    is_session_tombstoned,
)
from backend.src.slothbearflow_backend.memory.summary_memory import (
    cancel_summary_update,
)
from backend.src.slothbearflow_backend.mcp import get_mcp_status
from backend.src.slothbearflow_backend.observability import get_observability
from backend.src.slothbearflow_backend.observability.context import current_trace_id
from backend.src.slothbearflow_backend.observability.middleware import RequestTraceMiddleware
from backend.src.slothbearflow_backend.output_parser import structured_chat_output_from_text
from backend.src.slothbearflow_backend.output_schema import (
    ChatOutput,
    Citation,
    ToolTraceOutput,
)
from backend.src.slothbearflow_backend.persistence.postgres import postgres_persistence
from backend.src.slothbearflow_backend.rag.milvus_store import get_vector_store, get_vector_store_status
from backend.src.slothbearflow_backend.rag.security import (
    begin_citation_recall,
    citation_source_is_safe,
    clear_citation_recall,
)
from backend.src.slothbearflow_backend.tools.rag_tool import (
    get_last_rag_citations,
    get_last_rag_sources,
)
from backend.src.slothbearflow_backend.worker.background import worker_loop
from backend.src.slothbearflow_backend.security.approval import (
    ApprovalStoreUnavailable,
    approval_store,
)
from backend.src.slothbearflow_backend.security.audit import (
    audit_event,
    read_recent_audit_events,
    verify_audit_chain,
)
from backend.src.slothbearflow_backend.security.auth import (
    AUTH_COOKIE_NAME,
    authenticate_credentials,
    issue_access_token,
    namespace_session_id,
    require_scopes,
)
from backend.src.slothbearflow_backend.security.identity import Principal
from backend.src.slothbearflow_backend.security.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_distributed_rate_limit,
)
from backend.src.slothbearflow_backend.security.request_guard import (
    LocalAuthBoundaryMiddleware,
    RequestSizeLimitMiddleware,
)
from backend.src.slothbearflow_backend.security.turn_state import cancel_turn, end_turn


_login_semaphore = threading.BoundedSemaphore(value=4)


def _configure_logging() -> None:
    settings = get_settings()
    log_dir = os.path.abspath(settings.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    app_log_path = os.path.join(log_dir, settings.app_log_file)
    access_log_path = os.path.join(log_dir, settings.access_log_file)
    error_log_path = os.path.join(log_dir, settings.error_log_file)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    access_formatter = AccessFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(client_addr)s - "%(request_line)s" %(status_code)s'
    )
    def ensure_file_handler(
        target_logger: logging.Logger,
        file_path: str,
        *,
        level: int,
        formatter: logging.Formatter,
    ) -> None:
        if any(
            isinstance(handler, logging.FileHandler)
            and os.path.abspath(getattr(handler, "baseFilename", "")) == file_path
            for handler in target_logger.handlers
        ):
            return
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        target_logger.addHandler(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
    ensure_file_handler(
        root_logger,
        app_log_path,
        level=log_level,
        formatter=formatter,
    )

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(log_level)
    ensure_file_handler(
        uvicorn_logger,
        error_log_path,
        level=log_level,
        formatter=formatter,
    )

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(log_level)
    ensure_file_handler(
        uvicorn_error_logger,
        error_log_path,
        level=log_level,
        formatter=formatter,
    )

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(log_level)
    ensure_file_handler(
        uvicorn_access_logger,
        access_log_path,
        level=log_level,
        formatter=access_formatter,
    )


_configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.chat_semaphore = asyncio.Semaphore(settings.chat_concurrency_limit)
    observability = get_observability(settings)
    schema_ready = postgres_persistence.ensure_schema(settings)
    if (
        settings.app_env == "production"
        and settings.enable_postgres_persistence
        and not schema_ready
    ):
        raise RuntimeError("production PostgreSQL schema is not at the required Alembic head")
    # 初始化即建立工具白名单（tool guard，对标「初始化时建立白名单」）。
    try:
        from backend.src.slothbearflow_backend.security import get_tool_policy

        _tool_policy = get_tool_policy(settings)
        logger.info(
            "tool guard: mode=%s default_action=%s allow=%s",
            settings.tool_guard_mode,
            _tool_policy.default_action,
            _tool_policy.allowed_tool_names(),
        )
        if settings.app_env == "production" and _tool_policy.default_action != "deny":
            raise RuntimeError("production tool policy must use default_action=deny")
    except Exception:
        logger.exception("工具策略预热失败")
        if settings.app_env == "production":
            raise
    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.job_queue_max)
    app.state.job_queue = queue
    postgres_persistence.fail_unrecoverable_ingest_jobs(settings=settings)
    app.state.worker_task = asyncio.create_task(worker_loop(queue, settings))
    logger.info("后台 worker 已启动")
    yield
    app.state.worker_task.cancel()
    try:
        await app.state.worker_task
    except asyncio.CancelledError:
        pass
    postgres_persistence.close()
    observability.flush()
    logger.info("后台 worker 已停止")

_bootstrap_settings = get_settings()
app = FastAPI(
    title="SlothBearFlow Agent API",
    lifespan=lifespan,
    docs_url="/docs" if _bootstrap_settings.docs_enabled else None,
    redoc_url="/redoc" if _bootstrap_settings.docs_enabled else None,
)
app.add_middleware(RequestTraceMiddleware, settings=get_settings())
app.add_middleware(LocalAuthBoundaryMiddleware)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=get_settings().api_max_request_bytes,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_bootstrap_settings.cors_origins_json,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    message: str = Field(..., min_length=1)


class ChatResponse(ChatOutput):
    session_id: str
    raw_output: Optional[str] = None
    turn_id: str
    stop_reason: str
    steps: int = 0
    tool_trace: List[ToolTraceOutput] = Field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    executor: str = ""
    prompt_version: str = ""
    trace_id: str = ""


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(default="upload", max_length=256)
    visibility: str = Field(default="tenant", pattern=r"^(private|tenant|public)$")
    allowed_roles: List[str] = Field(default_factory=list, max_length=20)

    @field_validator("allowed_roles")
    @classmethod
    def validate_roles(cls, values: List[str]) -> List[str]:
        normalized = []
        for value in values:
            role = str(value or "").strip()
            if not role or len(role) > 64 or not role.replace("_", "-").isalnum():
                raise ValueError("allowed roles must be short alphanumeric identifiers")
            normalized.append(role)
        return sorted(set(normalized))

    @model_validator(mode="after")
    def validate_visibility_roles(self) -> "IngestRequest":
        if self.visibility == "public" and self.allowed_roles:
            raise ValueError("public knowledge cannot also declare allowed_roles")
        return self


class IngestResponse(BaseModel):
    accepted: bool = True
    job_id: str


class IngestStatusResponse(BaseModel):
    job_id: str
    source: str
    text_length: int
    status: str
    error_detail: str = ""


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=1024)


class LoginResponse(BaseModel):
    expires_in: int
    user: Dict[str, Any]


class ApprovalDecisionRequest(BaseModel):
    approve: bool


def _release_chat_slot(
    semaphore: asyncio.Semaphore,
    prepared: Any,
) -> None:
    execution_task = getattr(prepared, "execution_task", None)
    if execution_task is not None and not execution_task.done():
        def release_after_execution(task: asyncio.Task) -> None:
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
            finally:
                semaphore.release()

        execution_task.add_done_callback(release_after_execution)
        return
    semaphore.release()


def _release_chat_slot_after_task(
    semaphore: asyncio.Semaphore,
    task: asyncio.Task[Any],
) -> None:
    if task.done():
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
        semaphore.release()
        return

    def release(completed: asyncio.Task[Any]) -> None:
        try:
            completed.exception()
        except asyncio.CancelledError:
            pass
        finally:
            semaphore.release()

    task.add_done_callback(release)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "LangChain Prod Agent",
        "ok": True,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "chat": "/chat",
        "ingest": "/ingest",
        "auth": "/auth/login",
    }


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest, request: Request, response: Response) -> LoginResponse:
    settings = get_settings()
    client_host = str(request.client.host if request.client is not None else "unknown")
    try:
        for key, limit in (
            ("login:global", max(20, settings.rate_limit_per_minute)),
            ("login:ip:" + client_host, max(8, settings.rate_limit_per_minute // 3)),
            (
                "login:user:" + req.username.lower(),
                max(5, settings.rate_limit_per_minute // 4),
            ),
        ):
            check_distributed_rate_limit(key, limit=limit, settings=settings)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail="Too many login attempts.") from exc
    except RateLimitUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Authentication rate limiter is unavailable."
        ) from exc
    if not _login_semaphore.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Too many concurrent login attempts.")
    try:
        principal = authenticate_credentials(req.username, req.password, settings)
    finally:
        _login_semaphore.release()
    access_token = issue_access_token(principal, settings)
    response.set_cookie(
        AUTH_COOKIE_NAME,
        access_token,
        max_age=settings.auth_token_ttl_sec,
        httponly=True,
        secure=settings.app_env in {"staging", "production"},
        samesite="strict",
        path="/",
    )
    return LoginResponse(
        expires_in=settings.auth_token_ttl_sec,
        user={
            "user_id": principal.user_id,
            "username": principal.username,
            "tenant_id": principal.tenant_id,
            "roles": sorted(principal.roles),
            "scopes": sorted(principal.scopes),
        },
    )


@app.get("/auth/me")
def auth_me(
    principal: Principal = Depends(require_scopes()),
) -> Dict[str, Any]:
    return {
        "user_id": principal.user_id,
        "username": principal.username,
        "tenant_id": principal.tenant_id,
        "roles": sorted(principal.roles),
        "scopes": sorted(principal.scopes),
    }


@app.post("/auth/logout")
def auth_logout(
    response: Response,
) -> Dict[str, bool]:
    response.delete_cookie(
        AUTH_COOKIE_NAME,
        path="/",
        secure=get_settings().app_env in {"staging", "production"},
        httponly=True,
        samesite="strict",
    )
    return {"ok": True}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


def _collect_runtime_status() -> Dict[str, Any]:
    started_at = time.perf_counter()
    logger.info("health start")
    settings = get_settings()
    logger.info("health settings loaded in %.3fs", time.perf_counter() - started_at)

    redis_started_at = time.perf_counter()
    logger.info("health redis ping start")
    redis_ok, redis_err = ping_redis(settings)
    logger.info(
        "health redis ping done in %.3fs ok=%s error=%s",
        time.perf_counter() - redis_started_at,
        redis_ok,
        redis_err,
    )

    milvus_started_at = time.perf_counter()
    logger.info("health milvus status start")
    vs_status = get_vector_store_status(settings)
    logger.info(
        "health milvus status done in %.3fs status=%s",
        time.perf_counter() - milvus_started_at,
        vs_status,
    )

    session_started_at = time.perf_counter()
    logger.info("health session load start")
    payload, client = get_redis_session("health-check", settings=settings)
    session_backend = "memory" if isinstance(client, InMemoryRedis) else "redis"
    effective_redis_ok = bool(redis_ok and session_backend == "redis")
    if not effective_redis_ok and not redis_err:
        redis_err = "using in-memory fallback"
    logger.info(
        "health session load done in %.3fs backend=%s messages=%s",
        time.perf_counter() - session_started_at,
        session_backend,
        len(payload.get("messages") or []),
    )

    postgres_status = postgres_persistence.get_status(settings)
    llm_status = get_llm_status(settings)
    rag_configured = bool(settings.use_rag and not settings.skip_milvus)
    postgres_required = bool(settings.enable_postgres_persistence)
    degraded = (
        not effective_redis_ok
        or (rag_configured and not bool(vs_status.get("enabled")))
        or (
            postgres_required
            and not bool(postgres_status.get("ready"))
        )
        or (
            bool(llm_status.get("checked"))
            and not bool(llm_status.get("ready"))
        )
    )
    supports_tools = llm_supports_tools(settings)
    executor = "basic"
    if supports_tools:
        executor = (
            "explicit_react"
            if settings.enable_explicit_react_runtime
            else "tool_calling"
        )

    response = {
        # `ok` remains the backwards-compatible liveness signal. `status`
        # communicates whether configured dependencies are fully available.
        "ok": True,
        "status": "degraded" if degraded else "ready",
        "redis": {"ok": effective_redis_ok, "error": redis_err},
        "session_store": {
            "backend": session_backend,
            "loaded_messages": len(payload.get("messages") or []),
        },
        "milvus": vs_status,
        "postgres_persistence": postgres_status,
        "llm": llm_status,
        "embedding": {
            "provider": get_embedding_provider(settings),
            "model": get_embedding_model_name(settings),
        },
        "capabilities": {
            "agent": {
                "executor": executor,
                "tool_calling": supports_tools,
                "streaming": settings.stream_output,
                "stream_format": settings.stream_output_format,
                "structured_output": settings.structured_output,
                "timeout_sec": settings.agent_timeout_sec,
                "prompt_version": settings.agent_prompt_version,
                "exact_tool_trace": True,
            },
            "security": {
                "tool_guard_mode": settings.tool_guard_mode,
                "output_scrubbing": settings.tool_scrub_output,
                "max_tool_calls_per_turn": settings.max_tool_calls_per_turn,
                "approval_mode": "one_time_human_approval",
                "auth_required": settings.auth_required,
                "rate_limit_per_minute": settings.rate_limit_per_minute,
                "chat_concurrency_limit": settings.chat_concurrency_limit,
                "audit_enabled": settings.audit_enabled,
            },
            "rag": {
                "enabled": rag_configured,
                "available": bool(vs_status.get("enabled")),
                "hybrid_retrieval": True,
            },
            "memory": {
                "session_backend": session_backend,
                "window_pairs": settings.memory_window_pairs,
                "summary_enabled": settings.async_summary_update,
                "postgres_restore": settings.postgres_restore_on_redis_miss,
            },
            "learning": {
                "background_review": settings.enable_background_review,
                "prompt_injection": settings.inject_learning_into_prompt,
            },
            "mcp": get_mcp_status(settings),
            "observability": get_observability(settings).status(),
        },
    }
    logger.info("health done in %.3fs", time.perf_counter() - started_at)
    return response


@app.get("/health")
def health() -> Dict[str, Any]:
    settings = get_settings()
    if settings.app_env in {"staging", "production"}:
        return {
            "ok": True,
            "status": "ready",
            "check": "liveness",
            "capabilities": {
                "security": {"auth_required": settings.auth_required}
            },
        }
    return _collect_runtime_status()


@app.get("/runtime/status")
def runtime_status(
    _principal: Principal = Depends(require_scopes("observability:read")),
) -> Dict[str, Any]:
    return _collect_runtime_status()


@app.get("/ready")
def readiness(response: Response) -> Dict[str, Any]:
    settings = get_settings()
    payload = _collect_runtime_status()
    ready = payload.get("status") == "ready"
    if not ready:
        response.status_code = 503
    return {
        "ok": ready,
        "status": payload.get("status"),
        "checks": {
            "redis": bool((payload.get("redis") or {}).get("ok")),
            "milvus": bool(
                settings.skip_milvus
                or not settings.use_rag
                or (payload.get("milvus") or {}).get("enabled")
            ),
            "postgres": bool(
                not settings.enable_postgres_persistence
                or (payload.get("postgres_persistence") or {}).get("ready")
            ),
            "llm": bool(
                not (payload.get("llm") or {}).get("checked")
                or (payload.get("llm") or {}).get("ready")
            ),
        },
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    request: Request,
    principal: Principal = Depends(require_scopes("knowledge:write")),
) -> IngestResponse:
    settings = get_settings()
    if not citation_source_is_safe(req.source):
        raise HTTPException(status_code=422, detail="Knowledge source metadata is unsafe.")
    if len(req.text) > settings.ingest_text_max_chars:
        raise HTTPException(status_code=413, detail="Knowledge text is too large.")
    _check_rate_limit(principal, "ingest", settings)
    if settings.skip_milvus or not settings.use_rag:
        raise HTTPException(
            status_code=400,
            detail="当前配置关闭了向量库写入（SKIP_MILVUS / USE_RAG）。",
        )
    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = request.app.state.job_queue
    job_payload = {
        "type": "ingest",
        "text": req.text,
        "source": req.source,
        "job_id": job_id,
        "metadata": {
            "tenant_id": principal.tenant_id,
            "owner_id": principal.user_id,
            "visibility": req.visibility,
            "allowed_roles": sorted(set(req.allowed_roles)),
        },
    }
    persistence_enabled = postgres_persistence.is_enabled(settings)
    if persistence_enabled:
        persisted = postgres_persistence.persist_ingest_job(
            job_id=job_id,
            source=req.source,
            text_length=len(req.text),
            status="queued",
            tenant_id=principal.tenant_id,
            owner_id=principal.user_id,
            payload=job_payload,
            settings=settings,
        )
        if not persisted:
            raise HTTPException(
                status_code=503,
                detail="Knowledge ingestion queue is temporarily unavailable.",
            )
        try:
            queue.put_nowait({"type": "ingest_ref", "job_id": job_id})
        except asyncio.QueueFull:
            # The durable worker polls queued jobs, so accepted work is not lost.
            logger.info("in-memory worker queue full; durable ingest remains queued: %s", job_id)
    else:
        try:
            queue.put_nowait(job_payload)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="任务队列已满，请稍后重试。")
    audit_event(
        settings,
        "knowledge.ingest_queued",
        actor=principal.user_id,
        tenant_id=principal.tenant_id,
        target=job_id,
        metadata={
            "source": req.source,
            "text_length": len(req.text),
            "visibility": req.visibility,
        },
    )
    return IngestResponse(accepted=True, job_id=job_id)


@app.get("/ingest/{job_id}", response_model=IngestStatusResponse)
def ingest_status(
    job_id: str = Path(..., min_length=1, max_length=128),
    principal: Principal = Depends(require_scopes("knowledge:read")),
) -> IngestStatusResponse:
    settings = get_settings()
    if not postgres_persistence.is_enabled(settings):
        raise HTTPException(
            status_code=503,
            detail="当前配置未启用 ingest 任务状态持久化。",
        )
    job = postgres_persistence.get_ingest_job(job_id, settings=settings)
    if job is None:
        if not postgres_persistence.get_status(settings).get("ready"):
            raise HTTPException(
                status_code=503,
                detail="Ingest status storage is temporarily unavailable.",
            )
        raise HTTPException(status_code=404, detail="未找到该 ingest 任务。")
    if settings.auth_required and str(job.get("tenant_id") or "") != principal.tenant_id:
        raise HTTPException(status_code=404, detail="未找到该 ingest 任务。")
    if (
        settings.auth_required
        and str(job.get("owner_id") or "") != principal.user_id
        and "admin" not in principal.roles
    ):
        raise HTTPException(status_code=404, detail="未找到该 ingest 任务。")
    return IngestStatusResponse(**job)


@app.get("/metrics", include_in_schema=False)
def metrics(
    request: Request,
) -> Response:
    settings = get_settings()
    if not settings.observability_enabled or not settings.prometheus_enabled:
        raise HTTPException(status_code=404, detail="Prometheus metrics are disabled.")
    expected = settings.metrics_bearer_token
    authorization = request.headers.get("authorization", "").strip()
    supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if expected:
        import hmac

        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Invalid metrics token.")
    elif settings.auth_required:
        raise HTTPException(
            status_code=503,
            detail="Metrics token is required when authentication is enabled.",
        )
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="prometheus-client is not installed.") from exc
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/observability/traces")
def recent_traces(
    limit: int = 30,
    principal: Principal = Depends(require_scopes("observability:read")),
) -> Dict[str, Any]:
    return {
        "items": get_observability(get_settings()).recent_traces(
            limit,
            tenant_id=principal.tenant_id,
            user_id="" if "admin" in principal.roles else principal.user_id,
        )
    }


@app.get("/observability/traces/{trace_id}")
def trace_detail(
    trace_id: str = Path(..., min_length=8, max_length=128),
    principal: Principal = Depends(require_scopes("observability:read")),
) -> Dict[str, Any]:
    trace = get_observability(get_settings()).get_trace(
        trace_id,
        tenant_id=principal.tenant_id,
        user_id="" if "admin" in principal.roles else principal.user_id,
    )
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found.")
    return trace


@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    principal: Principal = Depends(require_scopes("chat:write")),
) -> ChatResponse:
    started_at = time.perf_counter()
    settings = get_settings()
    if len(req.message) > settings.chat_message_max_chars:
        raise HTTPException(status_code=413, detail="Chat message is too large.")
    _check_rate_limit(principal, "chat", settings)
    semaphore: asyncio.Semaphore = request.app.state.chat_semaphore
    try:
        await asyncio.wait_for(
            semaphore.acquire(), timeout=settings.concurrency_wait_sec
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat concurrency limit reached; retry later.",
        ) from exc
    turn_input: Optional[TurnInput] = None
    try:
        storage_session_id = namespace_session_id(req.session_id, principal, settings)
        if postgres_persistence.is_enabled(settings):
            persistent_tombstone = postgres_persistence.is_session_tombstoned(
                storage_session_id, settings=settings
            )
            if persistent_tombstone is None and settings.app_env == "production":
                raise HTTPException(
                    status_code=503,
                    detail="Persistent memory is temporarily unavailable.",
                )
            if persistent_tombstone:
                raise HTTPException(
                    status_code=409,
                    detail="This session was deleted. Start a new session_id.",
                )
        _, memory_client = get_redis_session(storage_session_id, settings=settings)
        if is_session_tombstoned(memory_client, storage_session_id):
            raise HTTPException(
                status_code=409,
                detail="This session was deleted. Start a new session_id.",
            )
        # Request-scoped collaborators preserve the existing monkeypatch boundary.
        runner = ChatTurnRunner(
            settings,
            request.app.state.job_queue,
            build_agent_executor=build_agent_executor,
            get_vector_store=get_vector_store,
            structured_chat_output_from_text=structured_chat_output_from_text,
            get_last_rag_sources=get_last_rag_sources,
            get_last_rag_citations=get_last_rag_citations,
        )
        deadline = time.monotonic() + settings.agent_timeout_sec
        turn_input = TurnInput(
            session_id=storage_session_id,
            display_session_id=req.session_id,
            message=req.message,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            roles=sorted(principal.roles),
        )
        begin_citation_recall(turn_input.turn_id)
    except HTTPException:
        semaphore.release()
        if turn_input is not None:
            clear_citation_recall(turn_input.turn_id)
        raise
    except Exception as exc:
        semaphore.release()
        if turn_input is not None:
            clear_citation_recall(turn_input.turn_id)
        logger.exception("Chat dependency preparation failed")
        raise HTTPException(
            status_code=503,
            detail="Chat dependencies are temporarily unavailable.",
        ) from exc
    except BaseException:
        semaphore.release()
        if turn_input is not None:
            clear_citation_recall(turn_input.turn_id)
        raise

    assert turn_input is not None
    try:
        prepared = await runner.prepare(
            turn_input,
            deadline=deadline,
        )
        audit_event(
            settings,
            "chat.turn_started",
            actor=principal.user_id,
            tenant_id=principal.tenant_id,
            target=prepared.turn.turn_id,
            metadata={"message_length": len(req.message)},
        )
    except AgentPreparationTimeout as exc:
        cancel_turn("agent preparation timed out")
        end_turn()
        clear_citation_recall(turn_input.turn_id)
        _release_chat_slot_after_task(semaphore, exc.task)
        raise HTTPException(
            status_code=504,
            detail="Agent preparation timed out.",
        ) from exc
    except BaseException:
        cancel_turn("agent preparation failed")
        end_turn()
        clear_citation_recall(turn_input.turn_id)
        semaphore.release()
        raise

    if prepared.should_stream:
        media_type = (
            "text/event-stream" if prepared.stream_format == "sse" else "text/plain"
        )
        async def stream_with_release():
            try:
                async for item in runner.iter_stream(prepared):
                    yield item
            finally:
                _release_chat_slot(semaphore, prepared)
                clear_citation_recall(turn_input.turn_id)

        return StreamingResponse(stream_with_release(), media_type=media_type)

    try:
        result = await runner.run_blocking(prepared)
    finally:
        _release_chat_slot(semaphore, prepared)
        clear_citation_recall(turn_input.turn_id)
    response = ChatResponse(
        answer=result.answer,
        source=result.source,
        citations=result.citations,
        tools_used=result.tools_used,
        session_id=result.session_id,
        raw_output=result.raw_output,
        turn_id=result.turn_id,
        stop_reason=result.stop_reason,
        steps=result.steps,
        tool_trace=[ToolTraceOutput(**item) for item in result.tool_trace],
        latency_ms=result.latency_ms,
        model=result.model,
        executor=result.executor,
        prompt_version=result.prompt_version,
        trace_id=current_trace_id(),
    )
    logger.info("chat done in %.3fs", time.perf_counter() - started_at)
    return response


@app.get("/memory/{session_id}")
def export_memory(
    session_id: str = Path(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
    principal: Principal = Depends(require_scopes("memory:read")),
) -> Dict[str, Any]:
    settings = get_settings()
    storage_session_id = namespace_session_id(session_id, principal, settings)
    payload, _ = get_redis_session(storage_session_id, settings=settings)
    return {"session_id": session_id, "memory": payload}


@app.get("/knowledge/documents")
def knowledge_documents(
    limit: int = 100,
    principal: Principal = Depends(require_scopes("knowledge:read")),
) -> Dict[str, Any]:
    return {
        "items": postgres_persistence.list_knowledge_manifests(
            principal.tenant_id,
            user_id=principal.user_id,
            roles=principal.roles,
            is_admin="admin" in principal.roles,
            limit=limit,
            settings=get_settings(),
        )
    }


@app.delete("/memory/{session_id}")
def delete_memory(
    session_id: str = Path(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    ),
    principal: Principal = Depends(require_scopes("memory:delete")),
) -> Dict[str, Any]:
    settings = get_settings()
    storage_session_id = namespace_session_id(session_id, principal, settings)
    try:
        payload, client = get_redis_session(
            storage_session_id,
            settings=settings,
            allow_fallback=False,
            force_probe=True,
        )
        redis_generation = get_redis_session_generation(client, storage_session_id)
    except Exception as exc:  # noqa: BLE001
        audit_event(
            settings,
            "memory.delete_failed",
            actor=principal.user_id,
            tenant_id=principal.tenant_id,
            target=session_id,
            metadata={"stage": "read_state"},
        )
        raise HTTPException(
            status_code=503,
            detail="Memory deletion state is temporarily unavailable.",
        ) from exc

    persistent_state = postgres_persistence.get_session_state(
        storage_session_id,
        settings=settings,
    )
    if persistent_state is not None:
        current_generation = max(
            int(payload.get("generation") or 0),
            int(persistent_state.get("generation") or 0),
            redis_generation,
        )
        already_tombstoned = bool(persistent_state.get("tombstoned")) or bool(
            is_session_tombstoned(client, storage_session_id)
        )
        target_generation: Optional[int] = (
            current_generation if already_tombstoned else current_generation + 1
        )
    else:
        target_generation = None
    cancel_summary_update(storage_session_id)
    postgres_deleted = postgres_persistence.delete_session(
        storage_session_id,
        target_generation=target_generation,
        settings=settings,
    )
    if postgres_deleted is None:
        audit_event(
            settings,
            "memory.delete_failed",
            actor=principal.user_id,
            tenant_id=principal.tenant_id,
            target=session_id,
            metadata={
                "postgres_deleted": False,
                "redis_delete_attempted": False,
            },
        )
        raise HTTPException(
            status_code=503,
            detail="Persistent memory deletion failed; retry the request.",
        )
    if target_generation is None:
        persistent_state = postgres_persistence.get_session_state(
            storage_session_id,
            settings=settings,
        )
        if persistent_state is None or not bool(persistent_state.get("tombstoned")):
            audit_event(
                settings,
                "memory.delete_failed",
                actor=principal.user_id,
                tenant_id=principal.tenant_id,
                target=session_id,
                metadata={"stage": "confirm_generation"},
            )
            raise HTTPException(
                status_code=503,
                detail="Memory deletion generation could not be confirmed.",
            )
        target_generation = int(persistent_state.get("generation") or 0)
        already_tombstoned = True
        current_generation = target_generation
    deleted_generation = max(
        0,
        target_generation - 1 if already_tombstoned else current_generation,
    )
    try:
        redis_deleted = delete_session_payload(
            client,
            storage_session_id,
            generation=target_generation,
        )
    except Exception as exc:  # noqa: BLE001
        audit_event(
            settings,
            "memory.delete_failed",
            actor=principal.user_id,
            tenant_id=principal.tenant_id,
            target=session_id,
            metadata={"stage": "redis", "postgres_deleted": postgres_deleted},
        )
        raise HTTPException(
            status_code=503,
            detail="Redis memory deletion failed; retry the request.",
        ) from exc
    try:
        learning_deleted = LearningStore(
            learning_dir_for(settings, principal.tenant_id, principal.user_id)
        ).delete_by_source(
            tenant_id=principal.tenant_id,
            session_id=storage_session_id,
            generation=deleted_generation,
        )
    except Exception as exc:  # noqa: BLE001
        audit_event(
            settings,
            "memory.delete_failed",
            actor=principal.user_id,
            tenant_id=principal.tenant_id,
            target=session_id,
            metadata={"stage": "learning_cascade"},
        )
        raise HTTPException(
            status_code=503,
            detail="Derived memory deletion failed; retry the request.",
        ) from exc
    audit_event(
        settings,
        "memory.deleted",
        actor=principal.user_id,
        tenant_id=principal.tenant_id,
        target=session_id,
        metadata={
            "redis_deleted": redis_deleted,
            "postgres_deleted": postgres_deleted,
            "learning_deleted": learning_deleted,
            "generation": deleted_generation,
        },
    )
    return {"deleted": bool(redis_deleted or postgres_deleted), "session_id": session_id}


@app.get("/security/approvals")
def list_approvals(
    limit: int = 100,
    principal: Principal = Depends(require_scopes("security:read")),
) -> Dict[str, Any]:
    try:
        items = approval_store.list(principal, limit=limit, settings=get_settings())
    except ApprovalStoreUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Approval service is temporarily unavailable."
        ) from exc
    return {"items": items}


@app.post("/security/approvals/{approval_id}")
def decide_approval(
    req: ApprovalDecisionRequest,
    approval_id: str = Path(..., min_length=8, max_length=128),
    principal: Principal = Depends(require_scopes("security:approve")),
) -> Dict[str, Any]:
    try:
        row = approval_store.decide(
            approval_id,
            approve=req.approve,
            actor=principal,
            settings=get_settings(),
        )
    except ApprovalStoreUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Approval service is temporarily unavailable."
        ) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="Approval not found.")
    return row


@app.get("/security/audit")
def security_audit(
    limit: int = 100,
    principal: Principal = Depends(require_scopes("security:read")),
) -> Dict[str, Any]:
    items = read_recent_audit_events(get_settings(), limit=min(500, limit * 5))
    return {
        "items": [
            item for item in items if item.get("tenant_id") == principal.tenant_id
        ][:limit],
        "chain": verify_audit_chain(get_settings()),
    }


def _check_rate_limit(principal: Principal, operation: str, settings: Any) -> None:
    try:
        check_distributed_rate_limit(
            f"{principal.tenant_id}:{principal.user_id}:{operation}",
            limit=settings.rate_limit_per_minute,
            settings=settings,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.") from exc
    except RateLimitUnavailable as exc:
        raise HTTPException(
            status_code=503, detail="Rate limiter is temporarily unavailable."
        ) from exc
