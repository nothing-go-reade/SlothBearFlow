from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from uvicorn.logging import AccessFormatter

from app import build_agent_executor, get_settings
from app.llm import get_llm_model_name
from app.rag.embedding import get_embedding_model_name, get_embedding_provider
from app.deps import InMemoryRedis, ping_redis
from app.memory.redis_memory import (
    append_turn_and_save,
    get_redis_session,
    messages_from_payload,
)
from app.memory.short_memory import trim_message_window
from app.memory.summary_memory import enqueue_summary_update
from app.output_parser import structured_chat_output_from_text
from app.output_schema import ChatOutput, Citation
from app.persistence.postgres import postgres_persistence
from app.rag.milvus_store import get_vector_store, get_vector_store_status
from app.tools.rag_tool import get_last_rag_citations, get_last_rag_sources, reset_rag_sources
from app.worker.background import worker_loop


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
    access_file_formatter = logging.Formatter(
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
        formatter=access_file_formatter,
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


def _detect_used_tools(raw_output: str, has_citations: bool) -> list[str]:
    tools: list[str] = []
    raw_lower = raw_output.lower()
    if "weather" in raw_lower or "天气查询结果" in raw_output:
        tools.append("get_weather")
    if "最近会话上下文" in raw_output:
        tools.append("get_session_context")
    if has_citations:
        tools.append("search_knowledge")
    return tools


def _should_stream_response(settings: Any, executor: Any) -> tuple[bool, str]:
    if not settings.stream_output:
        return False, "stream_output_disabled"
    if settings.structured_output:
        return False, "structured_output_enabled"
    if not hasattr(executor, "stream"):
        return False, "executor_not_streamable"
    return True, "enabled"


def _normalize_stream_output_format(value: str) -> str:
    normalized = str(value or "plain").strip().lower()
    return normalized if normalized in {"plain", "sse"} else "plain"


_STREAM_DONE = object()


def _next_stream_chunk(iterator: Any) -> Any:
    return next(iterator, _STREAM_DONE)


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
    logger.info(
        "chat start session_id=%s message_length=%s",
        req.session_id,
        len(req.message),
    )
    settings = get_settings()
    logger.info("chat settings loaded in %.3fs", time.perf_counter() - started_at)

    reset_started_at = time.perf_counter()
    reset_rag_sources()
    logger.info("chat rag context reset in %.3fs", time.perf_counter() - reset_started_at)

    session_started_at = time.perf_counter()
    logger.info("chat session load start session_id=%s", req.session_id)
    payload, client = get_redis_session(req.session_id, settings=settings)
    logger.info(
        "chat session load done in %.3fs backend=%s messages=%s summary_chars=%s",
        time.perf_counter() - session_started_at,
        "memory" if isinstance(client, InMemoryRedis) else "redis",
        len(payload.get("messages") or []),
        len(str(payload.get("summary") or "")),
    )

    history_started_at = time.perf_counter()
    history = messages_from_payload(list(payload.get("messages") or []))
    windowed = trim_message_window(history, settings.memory_window_pairs)
    logger.info(
        "chat history prepared in %.3fs total_messages=%s windowed_messages=%s",
        time.perf_counter() - history_started_at,
        len(history),
        len(windowed),
    )

    vs_started_at = time.perf_counter()
    logger.info("chat vector store prepare start")
    vs = get_vector_store(settings)
    logger.info(
        "chat vector store prepare done in %.3fs enabled=%s",
        time.perf_counter() - vs_started_at,
        vs is not None,
    )

    executor_started_at = time.perf_counter()
    logger.info("chat executor build start")
    executor = build_agent_executor(
        vector_store=vs,
        chat_history=windowed,
        rolling_summary=str(payload.get("summary") or "") or None,
        settings=settings,
    )
    logger.info(
        "chat executor build done in %.3fs",
        time.perf_counter() - executor_started_at,
    )

    should_stream, stream_reason = _should_stream_response(settings, executor)
    logger.info(
        "chat stream decision enabled=%s reason=%s",
        should_stream,
        stream_reason,
    )

    if should_stream:
        logger.info("chat streaming response start")
        stream_format = _normalize_stream_output_format(settings.stream_output_format)
        logger.info("chat streaming response format=%s", stream_format)

        async def event_stream():
            stream_started_at = time.perf_counter()
            payload_input = {"input": req.message, "chat_history": windowed}
            full_output_parts: list[str] = []
            if stream_format == "sse":
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "start", "session_id": req.session_id},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            iterator = executor.stream(payload_input)
            while True:
                chunk = await asyncio.to_thread(_next_stream_chunk, iterator)
                if chunk is _STREAM_DONE:
                    break
                text = str(chunk.get("output") or "")
                if not text:
                    continue
                full_output_parts.append(text)
                if stream_format == "sse":
                    yield (
                        "data: "
                        + json.dumps(
                            {"type": "chunk", "content": text}, ensure_ascii=False
                        )
                        + "\n\n"
                    )
                else:
                    yield text

            answer = "".join(full_output_parts)
            append_turn_and_save(
                client,
                req.session_id,
                payload,
                req.message,
                answer,
            )
            postgres_persistence.persist_chat_turn(
                session_id=req.session_id,
                user_message=req.message,
                assistant_message=answer,
                raw_output=answer,
                response_source="agent",
                tools_used=[],
                citations=[],
                settings=settings,
            )
            if settings.async_summary_update:
                queue: asyncio.Queue = request.app.state.job_queue
                await enqueue_summary_update(queue, req.session_id)
            logger.info(
                "chat streaming response done in %.3fs answer_chars=%s",
                time.perf_counter() - stream_started_at,
                len(answer),
            )
            if stream_format == "sse":
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "session_id": req.session_id,
                            "answer": answer,
                            "source": "agent",
                            "citations": [],
                            "tools_used": [],
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

        media_type = "text/event-stream" if stream_format == "sse" else "text/plain"
        return StreamingResponse(event_stream(), media_type=media_type)

    try:
        invoke_started_at = time.perf_counter()
        logger.info("chat agent invoke start")
        result = await asyncio.to_thread(
            executor.invoke,
            {"input": req.message, "chat_history": windowed},
        )
        logger.info(
            "chat agent invoke done in %.3fs result_keys=%s",
            time.perf_counter() - invoke_started_at,
            sorted(result.keys()),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent 调用失败")
        raise HTTPException(status_code=502, detail=str(e)) from e

    postprocess_started_at = time.perf_counter()
    raw = str(result.get("output") or "")
    rag_sources = get_last_rag_sources()
    rag_citations = [Citation(**item) for item in get_last_rag_citations()]
    rag_hint = ",".join(sorted(set(rag_sources))) if rag_sources else ""
    tools_used = _detect_used_tools(raw, bool(rag_citations))
    logger.info(
        "chat result parsed in %.3fs raw_chars=%s citations=%s tools=%s",
        time.perf_counter() - postprocess_started_at,
        len(raw),
        len(rag_citations),
        tools_used,
    )

    if settings.structured_output:
        try:
            structured_started_at = time.perf_counter()
            logger.info("chat structured output start")
            structured = await asyncio.to_thread(
                structured_chat_output_from_text,
                raw,
                rag_hint=rag_hint,
                citations=rag_citations,
                tools_used=tools_used,
                settings=settings,
            )
            logger.info(
                "chat structured output done in %.3fs answer_chars=%s",
                time.perf_counter() - structured_started_at,
                len(structured.answer or ""),
            )
        except Exception:
            logger.exception("结构化输出失败，回退为原文本")
            structured = ChatOutput(
                answer=raw,
                source=rag_hint or "agent",
                citations=rag_citations,
                tools_used=tools_used,
            )
    else:
        structured = ChatOutput(
            answer=raw,
            source=rag_hint or "agent",
            citations=rag_citations,
            tools_used=tools_used,
        )

    save_started_at = time.perf_counter()
    logger.info("chat session save start")
    append_turn_and_save(
        client,
        req.session_id,
        payload,
        req.message,
        structured.answer,
    )
    postgres_persistence.persist_chat_turn(
        session_id=req.session_id,
        user_message=req.message,
        assistant_message=structured.answer,
        raw_output=raw,
        response_source=structured.source or rag_hint or "agent",
        tools_used=structured.tools_used,
        citations=[item.model_dump() for item in structured.citations],
        settings=settings,
    )
    logger.info("chat session save done in %.3fs", time.perf_counter() - save_started_at)

    if settings.async_summary_update:
        summary_started_at = time.perf_counter()
        logger.info("chat summary enqueue start")
        queue: asyncio.Queue = request.app.state.job_queue
        await enqueue_summary_update(queue, req.session_id)
        logger.info(
            "chat summary enqueue done in %.3fs",
            time.perf_counter() - summary_started_at,
        )

    response = ChatResponse(
        answer=structured.answer,
        source=structured.source or rag_hint or "",
        citations=structured.citations,
        tools_used=structured.tools_used,
        session_id=req.session_id,
        raw_output=raw,
    )
    logger.info("chat done in %.3fs", time.perf_counter() - started_at)
    return response
