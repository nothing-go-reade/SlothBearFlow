from __future__ import annotations

import asyncio
import logging
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from backend.src.slothbearflow_backend import Settings, get_chat_llm, get_settings
from backend.src.slothbearflow_backend.deps import get_redis
from backend.src.slothbearflow_backend.memory.redis_memory import (
    load_session_payload,
    messages_from_payload,
    update_summary,
)
from backend.src.slothbearflow_backend.persistence.postgres import postgres_persistence

logger = logging.getLogger(__name__)


async def enqueue_summary_update(queue: asyncio.Queue, session_id: str) -> None:
    await queue.put({"type": "summarize", "session_id": session_id})


def run_summary_job(session_id: str, settings: Optional[Settings] = None) -> None:
    """后台任务：压缩多轮对话为短摘要，写入 Redis（轻量占位实现）。"""
    settings = settings or get_settings()
    client = get_redis(settings)
    payload = load_session_payload(client, session_id)
    msgs = messages_from_payload(list(payload.get("messages") or []))
    if not msgs:
        return
    transcript = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content}"
        for m in msgs[-20:]
    )
    llm = get_chat_llm(settings, temperature=0.2)
    summary = llm.invoke(
        "请将以下对话压缩为不超过 120 字的中文摘要，保留用户目标与关键约束：\n"
        f"{transcript}"
    )
    text = summary.content if hasattr(summary, "content") else str(summary)
    update_summary(client, session_id, payload, str(text).strip())
    postgres_persistence.persist_summary(session_id, str(text).strip(), settings=settings)
    logger.info("会话摘要已更新: session_id=%s", session_id)
