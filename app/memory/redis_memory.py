from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app import Settings, get_settings
from app.deps import get_redis

logger = logging.getLogger(__name__)

SESSION_PREFIX = "chat:session:"


def _key(session_id: str) -> str:
    return f"{SESSION_PREFIX}{session_id}"


def default_session_payload() -> Dict[str, Any]:
    return {"messages": [], "summary": ""}


def load_session_payload(client: Any, session_id: str) -> Dict[str, Any]:
    raw = client.get(_key(session_id))
    if not raw:
        return default_session_payload()
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return default_session_payload()
        data.setdefault("messages", [])
        data.setdefault("summary", "")
        if not isinstance(data["messages"], list):
            data["messages"] = []
        return data
    except json.JSONDecodeError:
        logger.warning("会话 JSON 损坏，已重置: %s", session_id)
        return default_session_payload()


def save_session_payload(
    client: Any,
    session_id: str,
    payload: Dict[str, Any],
    *,
    ttl_sec: int = 86400 * 7
) -> None:
    key = _key(session_id)
    client.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl_sec)


def messages_from_payload(rows: List[Dict[str, Any]]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            out.append(AIMessage(content=str(content)))
    return out


def payload_from_messages(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            rows.append({"role": "user", "content": str(m.content)})
        elif isinstance(m, AIMessage):
            rows.append({"role": "assistant", "content": str(m.content)})
    return rows


def get_redis_session(
    session_id: str,
    *,
    settings: Optional[Settings] = None,
) -> Tuple[Dict[str, Any], Any]:
    settings = settings or get_settings()
    client = get_redis(settings)
    payload = load_session_payload(client, session_id)
    return payload, client


def append_turn_and_save(
    client: Any,
    session_id: str,
    payload: Dict[str, Any],
    user_text: str,
    assistant_text: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = list(payload.get("messages") or [])
    rows.append({"role": "user", "content": user_text})
    rows.append({"role": "assistant", "content": assistant_text})
    payload["messages"] = rows
    save_session_payload(client, session_id, payload, ttl_sec=86400 * 7)
    return payload


def update_summary(
    client: Any, session_id: str, payload: Dict[str, Any], summary: str
) -> None:
    payload["summary"] = summary
    save_session_payload(client, session_id, payload)
