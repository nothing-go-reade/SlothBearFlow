from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, Dict, Optional

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.learning.review_agent import run_review_job
from backend.src.slothbearflow_backend.memory.summary_memory import (
    mark_summary_complete,
    run_summary_job,
)
from backend.src.slothbearflow_backend.persistence.postgres import postgres_persistence
from backend.src.slothbearflow_backend.rag.ingest import (
    build_ingest_metadata,
    ingest_plain_text_with_outcome,
)
from backend.src.slothbearflow_backend.rag.embedding import get_embedding_model_name
from backend.src.slothbearflow_backend.rag.splitter import CHUNKER_VERSION
from backend.src.slothbearflow_backend.rag.milvus_store import get_vector_store

logger = logging.getLogger(__name__)


async def _run_ingest_job(job: Dict[str, Any], settings: Settings) -> int:
    vs = get_vector_store(settings)
    job_id = str(job.get("job_id") or "")
    source = str(job.get("source") or "upload")
    text = str(job.get("text") or "")
    metadata = build_ingest_metadata(
        text,
        source=source,
        metadata=dict(job.get("metadata") or {}) | {"job_id": job_id},
        settings=settings,
    )
    persistence_enabled = postgres_persistence.is_enabled(settings)
    if vs is None:
        logger.error("ingest 延迟：向量库不可用")
        raise RuntimeError("vector_store_unavailable")

    if job_id and persistence_enabled and not postgres_persistence.persist_ingest_job(
        job_id=job_id,
        source=source,
        text_length=len(text),
        status="processing",
        tenant_id=str(metadata.get("tenant_id") or ""),
        owner_id=str(metadata.get("owner_id") or ""),
        settings=settings,
    ):
        raise RuntimeError("ingest outbox processing checkpoint failed")

    with postgres_persistence.document_ingest_lock(
        str(metadata["document_id"]),
        settings=settings,
    ):
        if job_id and persistence_enabled:
            superseded = postgres_persistence.is_document_ingest_superseded(
                str(metadata["document_id"]),
                job_id,
                settings=settings,
            )
            if superseded is None:
                raise RuntimeError("document ingest CAS state is unavailable")
            if superseded:
                if not postgres_persistence.persist_ingest_job(
                    job_id=job_id,
                    source=source,
                    text_length=len(text),
                    status="skipped",
                    error_detail="superseded_by_newer_document_version",
                    tenant_id=str(metadata.get("tenant_id") or ""),
                    owner_id=str(metadata.get("owner_id") or ""),
                    settings=settings,
                ):
                    raise RuntimeError("superseded ingest checkpoint failed")
                return 0

        ingest_task = asyncio.create_task(
            asyncio.to_thread(
                ingest_plain_text_with_outcome,
                text,
                source=source,
                vector_store=vs,
                settings=settings,
                metadata=metadata,
            )
        )
        try:
            outcome = await asyncio.shield(ingest_task)
        except asyncio.CancelledError:
            # A running thread cannot be force-cancelled. Keep the document lock
            # until the write/cleanup finishes so a replacement worker cannot
            # concurrently mutate the same document.
            logger.warning("worker 关闭中，等待当前 ingest 写入安全结束: job_id=%s", job_id)
            await asyncio.shield(ingest_task)
            raise
        if not outcome.cleanup_confirmed:
            raise RuntimeError("Milvus stale-version cleanup was not confirmed")
        if job_id and persistence_enabled and not postgres_persistence.record_ingest_checkpoint(
            job_id,
            milvus_cleanup_completed=True,
            settings=settings,
        ):
            raise RuntimeError("Milvus cleanup checkpoint persistence failed")

        manifest_persisted = postgres_persistence.persist_knowledge_manifest(
            document_id=str(metadata["document_id"]),
            document_version=str(metadata["document_version"]),
            job_id=job_id,
            source=source,
            tenant_id=str(metadata.get("tenant_id") or ""),
            owner_id=str(metadata.get("owner_id") or ""),
            visibility=str(metadata.get("visibility") or "tenant"),
            allowed_roles=metadata.get("allowed_roles") or [],
            chunk_count=outcome.chunk_count,
            chunker_version=CHUNKER_VERSION,
            embedding_model=get_embedding_model_name(settings),
            chunking_contract=dict(metadata.get("chunking_contract") or {}),
            settings=settings,
        )
        if persistence_enabled and not manifest_persisted:
            raise RuntimeError("knowledge manifest persistence failed")
        if job_id and persistence_enabled:
            if not postgres_persistence.record_ingest_checkpoint(
                job_id,
                manifest_completed=True,
                settings=settings,
            ):
                raise RuntimeError("manifest checkpoint persistence failed")
            if not postgres_persistence.complete_ingest_job(
                job_id,
                settings=settings,
            ):
                raise RuntimeError("ingest outbox completion preconditions failed")
    return outcome.chunk_count


async def _run_thread_with_timeout(
    function: Any,
    *args: Any,
    timeout_sec: float,
) -> Optional[asyncio.Task[Any]]:
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=max(0.1, timeout_sec))
        return None
    except asyncio.TimeoutError:
        return task


async def worker_loop(queue: asyncio.Queue, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    while True:
        from_queue = True
        try:
            job: Optional[Dict[str, Any]] = await asyncio.wait_for(
                queue.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            from_queue = False
            job = await asyncio.to_thread(
                postgres_persistence.claim_ingest_job,
                settings=settings,
            )
            if job is None:
                continue
        try:
            job_type = job.get("type")
            if job_type == "ingest_ref":
                claimed = await asyncio.to_thread(
                    postgres_persistence.claim_ingest_job,
                    str(job.get("job_id") or ""),
                    settings=settings,
                )
                if claimed is None:
                    continue
                job = claimed
                job_type = "ingest"
            if job_type == "ingest":
                await _run_ingest_job(job, settings)
            elif job_type == "summarize":
                sid = str(job.get("session_id") or "")
                if not sid:
                    continue
                late_task = await _run_thread_with_timeout(
                    run_summary_job,
                    sid,
                    settings,
                    int(job.get("generation") or 0),
                    timeout_sec=settings.summary_timeout_sec,
                )
                if late_task is not None:
                    logger.warning("摘要任务超时，后台执行仍受 tombstone 约束: %s", sid)
                    job["_late_summary_task"] = late_task
            elif job_type == "review":
                snapshot = job.get("snapshot") or {}
                late_task = await _run_thread_with_timeout(
                    run_review_job,
                    snapshot,
                    settings,
                    timeout_sec=settings.review_timeout_sec,
                )
                if late_task is not None:
                    logger.warning("后台复盘超时，worker 已继续处理其他任务")
                    late_task.add_done_callback(_consume_task_exception)
            else:
                logger.warning("未知任务类型: %s", job_type)
        except asyncio.CancelledError:
            if job.get("type") == "ingest" and job.get("job_id"):
                postgres_persistence.persist_ingest_job(
                    job_id=str(job.get("job_id")),
                    source=str(job.get("source") or "upload"),
                    text_length=len(str(job.get("text") or "")),
                    status="queued",
                    error_detail="worker_shutdown_requeued",
                    tenant_id=str((job.get("metadata") or {}).get("tenant_id") or ""),
                    owner_id=str((job.get("metadata") or {}).get("owner_id") or ""),
                    settings=settings,
                )
            raise
        except Exception as exc:
            if job.get("type") == "ingest" and job.get("job_id"):
                attempts = int(job.get("attempts") or 1)
                error_detail = (
                    "vector_store_unavailable"
                    if str(exc) == "vector_store_unavailable"
                    else "background_worker_error"
                )
                if attempts < settings.ingest_max_attempts:
                    postgres_persistence.defer_ingest_job(
                        str(job.get("job_id")),
                        error_detail=error_detail,
                        delay_sec=settings.ingest_retry_backoff_sec * (2 ** max(0, attempts - 1)),
                        settings=settings,
                    )
                else:
                    postgres_persistence.persist_ingest_job(
                        job_id=str(job.get("job_id")),
                        source=str(job.get("source") or "upload"),
                        text_length=len(str(job.get("text") or "")),
                        status="failed",
                        error_detail=error_detail,
                        tenant_id=str((job.get("metadata") or {}).get("tenant_id") or ""),
                        owner_id=str((job.get("metadata") or {}).get("owner_id") or ""),
                        settings=settings,
                    )
            logger.exception("后台任务失败: %s", job)
        finally:
            if job.get("type") == "summarize":
                late_summary_task = job.get("_late_summary_task")
                if isinstance(late_summary_task, asyncio.Task):
                    sid = str(job.get("session_id") or "")
                    late_summary_task.add_done_callback(
                        partial(_complete_late_summary, session_id=sid)
                    )
                else:
                    mark_summary_complete(str(job.get("session_id") or ""))
            if from_queue:
                queue.task_done()


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return
    if error is not None:
        logger.error(
            "后台超时任务最终失败",
            exc_info=(type(error), error, error.__traceback__),
        )


def _complete_late_summary(
    task: asyncio.Task[Any], *, session_id: str
) -> None:
    try:
        _consume_task_exception(task)
    finally:
        mark_summary_complete(session_id)
