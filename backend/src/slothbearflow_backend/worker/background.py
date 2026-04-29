from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from slothbearflow_backend import Settings, get_settings
from slothbearflow_backend.memory.summary_memory import run_summary_job
from slothbearflow_backend.persistence.postgres import postgres_persistence
from slothbearflow_backend.rag.ingest import ingest_plain_text
from slothbearflow_backend.rag.milvus_store import get_vector_store

logger = logging.getLogger(__name__)


async def worker_loop(queue: asyncio.Queue, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    while True:
        job: Dict[str, Any] = await queue.get()
        try:
            job_type = job.get("type")
            if job_type == "ingest":
                vs = get_vector_store(settings)
                job_id = str(job.get("job_id") or "")
                source = str(job.get("source") or "upload")
                text = str(job.get("text") or "")
                if vs is None:
                    logger.error("ingest 跳过：向量库不可用")
                    if job_id:
                        postgres_persistence.persist_ingest_job(
                            job_id=job_id,
                            source=source,
                            text_length=len(text),
                            status="skipped",
                            error_detail="vector_store_unavailable",
                            settings=settings,
                        )
                    continue
                await asyncio.to_thread(
                    ingest_plain_text,
                    text,
                    source=source,
                    vector_store=vs,
                    settings=settings,
                )
                if job_id:
                    postgres_persistence.persist_ingest_job(
                        job_id=job_id,
                        source=source,
                        text_length=len(text),
                        status="completed",
                        settings=settings,
                    )
            elif job_type == "summarize":
                sid = str(job.get("session_id") or "")
                if not sid:
                    continue
                await asyncio.to_thread(run_summary_job, sid, settings)
            else:
                logger.warning("未知任务类型: %s", job_type)
        except Exception:
            if job.get("type") == "ingest" and job.get("job_id"):
                postgres_persistence.persist_ingest_job(
                    job_id=str(job.get("job_id")),
                    source=str(job.get("source") or "upload"),
                    text_length=len(str(job.get("text") or "")),
                    status="failed",
                    error_detail="background_worker_error",
                    settings=settings,
                )
            logger.exception("后台任务失败: %s", job)
        finally:
            queue.task_done()
