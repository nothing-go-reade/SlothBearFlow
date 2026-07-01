from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uvicorn.logging import AccessFormatter

from backend.src.slothbearflow_backend import build_agent_executor, get_settings
from backend.src.slothbearflow_backend.agent.conversation_loop import ChatTurnRunner, TurnInput
from backend.src.slothbearflow_backend.llm import get_llm_model_name
from backend.src.slothbearflow_backend.rag.embedding import get_embedding_model_name, get_embedding_provider
from backend.src.slothbearflow_backend.deps import InMemoryRedis, ping_redis
from backend.src.slothbearflow_backend.memory.redis_memory import get_redis_session
from backend.src.slothbearflow_backend.output_parser import structured_chat_output_from_text
from backend.src.slothbearflow_backend.output_schema import ChatOutput, Citation
from backend.src.slothbearflow_backend.persistence.postgres import postgres_persistence
from backend.src.slothbearflow_backend.rag.milvus_store import get_vector_store, get_vector_store_status
from backend.src.slothbearflow_backend.tools.rag_tool import (
    get_last_rag_citations,
    get_last_rag_sources,
)
from backend.src.slothbearflow_backend.worker.background import worker_loop


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
    postgres_persistence.ensure_schema(settings)
    queue: asyncio.Queue = asyncio.Queue(maxsize=settings.job_queue_max)
    app.state.job_queue = queue
    app.state.worker_task = asyncio.create_task(worker_loop(queue, settings))
    logger.info("后台 worker 已启动")
    yield
    app.state.worker_task.cancel()
    try:
        await app.state.worker_task
    except asyncio.CancelledError:
        pass
    logger.info("后台 worker 已停止")

app = FastAPI(title="LangChain Prod Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)


class ChatResponse(ChatOutput):
    session_id: str
    raw_output: Optional[str] = None


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(default="upload", max_length=256)


class IngestResponse(BaseModel):
    accepted: bool = True
    job_id: str


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "LangChain Prod Agent",
        "ok": True,
        "docs": "/docs",
        "health": "/health",
        "chat": "/chat",
        "ingest": "/ingest",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> Dict[str, Any]:
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
    logger.info(
        "health session load done in %.3fs backend=%s messages=%s",
        time.perf_counter() - session_started_at,
        "memory" if isinstance(client, InMemoryRedis) else "redis",
        len(payload.get("messages") or []),
    )

    response = {
        "ok": True,
        "redis": {"ok": redis_ok, "error": redis_err},
        "session_store": {
            "backend": "memory" if isinstance(client, InMemoryRedis) else "redis",
            "loaded_messages": len(payload.get("messages") or []),
        },
        "milvus": vs_status,
        "postgres_persistence": postgres_persistence.get_status(settings),
        "llm": {
            "provider": settings.llm_provider,
            "model": get_llm_model_name(settings),
        },
        "embedding": {
            "provider": get_embedding_provider(settings),
            "model": get_embedding_model_name(settings),
        },
        "ollama_base_url": settings.ollama_base_url,
    }
    logger.info("health done in %.3fs", time.perf_counter() - started_at)
    return response


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    settings = get_settings()
    if settings.skip_milvus or not settings.use_rag:
        raise HTTPException(
            status_code=400,
            detail="当前配置关闭了向量库写入（SKIP_MILVUS / USE_RAG）。",
        )
    job_id = str(uuid.uuid4())
    queue: asyncio.Queue = request.app.state.job_queue
    try:
        queue.put_nowait(
            {"type": "ingest", "text": req.text, "source": req.source, "job_id": job_id}
        )
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="任务队列已满，请稍后重试。")
    postgres_persistence.persist_ingest_job(
        job_id=job_id,
        source=req.source,
        text_length=len(req.text),
        status="queued",
        settings=settings,
    )
    return IngestResponse(accepted=True, job_id=job_id)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    started_at = time.perf_counter()
    settings = get_settings()
    # 编排已抽到 ChatTurnRunner（对标 Hermes conversation_loop）。
    # 这里在请求期注入可被测试 monkeypatch 的协作者（按 main 模块解析），保留 patch 语义。
    runner = ChatTurnRunner(
        settings,
        request.app.state.job_queue,
        build_agent_executor=build_agent_executor,
        get_vector_store=get_vector_store,
        structured_chat_output_from_text=structured_chat_output_from_text,
        get_last_rag_sources=get_last_rag_sources,
        get_last_rag_citations=get_last_rag_citations,
    )
    prepared = await runner.prepare(
        TurnInput(session_id=req.session_id, message=req.message)
    )

    if prepared.should_stream:
        media_type = (
            "text/event-stream" if prepared.stream_format == "sse" else "text/plain"
        )
        return StreamingResponse(runner.iter_stream(prepared), media_type=media_type)

    result = await runner.run_blocking(prepared)
    response = ChatResponse(
        answer=result.answer,
        source=result.source,
        citations=result.citations,
        tools_used=result.tools_used,
        session_id=result.session_id,
        raw_output=result.raw_output,
    )
    logger.info("chat done in %.3fs", time.perf_counter() - started_at)
    return response
