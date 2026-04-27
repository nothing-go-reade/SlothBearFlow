from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from app import Settings, get_settings
from app.memory.summary_memory import run_summary_job
from app.rag.ingest import ingest_plain_text
from app.rag.milvus_store import get_vector_store

logger = logging.getLogger(__name__)


async def worker_loop(queue: asyncio.Queue, settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    while True:
        job: Dict[str, Any] = await queue.get()
        try:
            job_type = job.get("type")
            if job_type == "ingest":
                vs = get_vector_store(settings)
                if vs is None:
                    logger.error("ingest 跳过：向量库不可用")
                    continue
                text = str(job.get("text") or "")
                source = str(job.get("source") or "upload")
                await asyncio.to_thread(
                    ingest_plain_text,
                    text,
                    source=source,
                    vector_store=vs,
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
            logger.exception("后台任务失败: %s", job)
        finally:
            queue.task_done()
