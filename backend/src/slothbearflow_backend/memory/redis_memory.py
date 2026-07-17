from __future__ import annotations

import json
import logging
import threading
import time
import contextvars
from typing import Any, Dict, List, Optional, Tuple

import redis
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.deps import get_redis
from backend.src.slothbearflow_backend.persistence.postgres import postgres_persistence
from backend.src.slothbearflow_backend.memory.privacy import redact_memory_text

logger = logging.getLogger(__name__)

SESSION_PREFIX = "chat:session:"
TOMBSTONE_PREFIX = "chat:session:tombstone:"
EPOCH_PREFIX = "chat:session:epoch:"
DEFAULT_TOMBSTONE_TTL_SEC = 86400 * 30
_session_locks: Dict[str, threading.RLock] = {}
_session_locks_guard = threading.Lock()
_session_generation_ctx: contextvars.ContextVar[Tuple[str, int]] = (
    contextvars.ContextVar("memory_session_generation", default=("", 0))
)


class SessionTombstonedError(RuntimeError):
    pass


class SessionGenerationMismatchError(RuntimeError):
    pass


class SessionVersionMismatchError(RuntimeError):
    pass


def _key(session_id: str) -> str:
    return f"{SESSION_PREFIX}{session_id}"


def _tombstone_key(session_id: str) -> str:
    return f"{TOMBSTONE_PREFIX}{session_id}"


def _epoch_key(session_id: str) -> str:
    return f"{EPOCH_PREFIX}{session_id}"


def default_session_payload(generation: int = 0) -> Dict[str, Any]:
    return {
        "messages": [],
        "summary": "",
        "version": 0,
        "generation": max(0, int(generation)),
    }


def _decode_payload(raw: Any) -> Dict[str, Any]:
    if not raw:
        return default_session_payload()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default_session_payload()
    if not isinstance(data, dict):
        return default_session_payload()
    data.setdefault("messages", [])
    data.setdefault("summary", "")
    data.setdefault("version", 0)
    data.setdefault("generation", 0)
    if not isinstance(data["messages"], list):
        data["messages"] = []
    try:
        data["generation"] = max(0, int(data["generation"]))
    except (TypeError, ValueError):
        data["generation"] = 0
    return data


def _payload_is_invalid(raw: Any) -> bool:
    if not raw:
        return False
    try:
        return not isinstance(json.loads(raw), dict)
    except (json.JSONDecodeError, TypeError):
        return True


def bind_session_generation(session_id: str, generation: int) -> None:
    _session_generation_ctx.set((str(session_id), max(0, int(generation))))


def current_session_generation(session_id: str) -> Optional[int]:
    current_session_id, generation = _session_generation_ctx.get()
    if current_session_id != str(session_id):
        return None
    return generation


def get_redis_session_generation(client: Any, session_id: str) -> int:
    try:
        return max(0, int(client.get(_epoch_key(session_id)) or 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def _ensure_redis_generation(client: Any, session_id: str, generation: int) -> int:
    generation = max(0, int(generation))
    epoch_key = _epoch_key(session_id)
    if isinstance(client, redis.Redis):
        script = (
            "local current = tonumber(redis.call('get', KEYS[1]) or '0'); "
            "local requested = tonumber(ARGV[1]); "
            "if current < requested then redis.call('set', KEYS[1], requested); "
            "return requested; end; return current"
        )
        return int(client.eval(script, 1, epoch_key, generation))
    with _lock_for_session(session_id):
        current = get_redis_session_generation(client, session_id)
        if current < generation:
            client.set(epoch_key, str(generation))
            return generation
        return current


def load_session_payload(client: Any, session_id: str) -> Dict[str, Any]:
    if is_session_tombstoned(client, session_id):
        return default_session_payload()
    raw = client.get(_key(session_id))
    data = _decode_payload(raw)
    if _payload_is_invalid(raw):
        logger.warning("会话 JSON 损坏，已重置: %s", session_id)
    return data


def save_session_payload(
    client: Any,
    session_id: str,
    payload: Dict[str, Any],
    *,
    ttl_sec: int = 86400 * 7
) -> None:
    key = _key(session_id)
    if isinstance(client, redis.Redis):
        client.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl_sec)
        return
    with _lock_for_session(session_id):
        client.set(key, json.dumps(payload, ensure_ascii=False), ex=ttl_sec)


def atomic_update_session(
    client: Any,
    session_id: str,
    mutator: Any,
    *,
    ttl_sec: int,
    max_retries: int = 8,
    expected_generation: Optional[int] = None,
) -> Dict[str, Any]:
    key = _key(session_id)
    tombstone_key = _tombstone_key(session_id)
    epoch_key = _epoch_key(session_id)
    expected = (
        max(0, int(expected_generation))
        if expected_generation is not None
        else None
    )
    if isinstance(client, redis.Redis):
        for _ in range(max_retries):
            try:
                with client.pipeline() as pipe:
                    pipe.watch(key, tombstone_key, epoch_key)
                    if pipe.get(tombstone_key):
                        raise SessionTombstonedError(
                            "session has been deleted; clear its tombstone before reuse"
                        )
                    generation = int(pipe.get(epoch_key) or 0)
                    if expected is not None and generation != expected:
                        raise SessionGenerationMismatchError(
                            "session generation changed while the turn was running"
                        )
                    current = _decode_payload(pipe.get(key))
                    if int(current.get("generation") or 0) != generation:
                        if current.get("messages") or current.get("summary"):
                            raise SessionGenerationMismatchError(
                                "session payload belongs to a stale generation"
                            )
                        current["generation"] = generation
                    updated = mutator(current)
                    updated["generation"] = generation
                    pipe.multi()
                    pipe.set(
                        key,
                        json.dumps(updated, ensure_ascii=False),
                        ex=max(1, int(ttl_sec)),
                    )
                    pipe.execute()
                    return updated
            except redis.WatchError:
                continue
        raise RuntimeError("concurrent session update retry limit exceeded")

    lock = _lock_for_session(session_id)
    with lock:
        if is_session_tombstoned(client, session_id):
            raise SessionTombstonedError(
                "session has been deleted; clear its tombstone before reuse"
            )
        generation = get_redis_session_generation(client, session_id)
        if expected is not None and generation != expected:
            raise SessionGenerationMismatchError(
                "session generation changed while the turn was running"
            )
        current = load_session_payload(client, session_id)
        if int(current.get("generation") or 0) != generation:
            if current.get("messages") or current.get("summary"):
                raise SessionGenerationMismatchError(
                    "session payload belongs to a stale generation"
                )
            current["generation"] = generation
        updated = mutator(current)
        updated["generation"] = generation
        save_session_payload(
            client,
            session_id,
            updated,
            ttl_sec=max(1, int(ttl_sec)),
        )
        return updated


def _lock_for_session(session_id: str) -> threading.RLock:
    with _session_locks_guard:
        return _session_locks.setdefault(session_id, threading.RLock())


def is_session_tombstoned(client: Any, session_id: str) -> bool:
    try:
        return bool(client.get(_tombstone_key(session_id)))
    except AttributeError:
        return False


def clear_session_tombstone(client: Any, session_id: str) -> bool:
    return bool(client.delete(_tombstone_key(session_id)))


def _restore_session_if_unchanged(
    client: Any,
    session_id: str,
    payload: Dict[str, Any],
    *,
    expected_raw: Any,
    ttl_sec: int,
    max_retries: int = 8,
) -> Tuple[Dict[str, Any], bool]:
    key = _key(session_id)
    tombstone_key = _tombstone_key(session_id)
    encoded = json.dumps(payload, ensure_ascii=False)
    ttl_sec = max(1, int(ttl_sec))

    if isinstance(client, redis.Redis):
        for _ in range(max_retries):
            try:
                with client.pipeline() as pipe:
                    pipe.watch(key, tombstone_key)
                    if pipe.get(tombstone_key):
                        pipe.unwatch()
                        return default_session_payload(), False
                    current_raw = pipe.get(key)
                    if current_raw != expected_raw:
                        pipe.unwatch()
                        return _decode_payload(current_raw), False
                    pipe.multi()
                    pipe.set(key, encoded, ex=ttl_sec)
                    pipe.execute()
                    return payload, True
            except redis.WatchError:
                continue
        raise RuntimeError("concurrent session restore retry limit exceeded")

    with _lock_for_session(session_id):
        if is_session_tombstoned(client, session_id):
            return default_session_payload(), False
        current_raw = client.get(key)
        if current_raw != expected_raw:
            return _decode_payload(current_raw), False
        save_session_payload(client, session_id, payload, ttl_sec=ttl_sec)
        return payload, True


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
    allow_fallback: bool = True,
    force_probe: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    settings = settings or get_settings()
    if allow_fallback and not force_probe:
        client = get_redis(settings)
    else:
        client = get_redis(
            settings,
            allow_fallback=allow_fallback,
            force_probe=force_probe,
        )
    persistent_state = postgres_persistence.get_session_state(
        session_id,
        settings=settings,
    )
    if persistent_state is None:
        raise RuntimeError("persistent memory tombstone state is unavailable")
    persistent_generation = int(persistent_state.get("generation") or 0)
    generation = _ensure_redis_generation(
        client,
        session_id,
        persistent_generation,
    )
    bind_session_generation(session_id, generation)
    if bool(persistent_state.get("tombstoned")):
        mark_session_deleted(client, session_id, generation=generation)
        return default_session_payload(generation), client
    if is_session_tombstoned(client, session_id):
        return default_session_payload(generation), client
    raw = client.get(_key(session_id))
    payload = _decode_payload(raw)
    payload_generation = int(payload.get("generation") or 0)
    if raw and payload_generation != generation:
        logger.warning(
            "忽略旧代际会话缓存: session_id=%s payload_generation=%s generation=%s",
            session_id,
            payload_generation,
            generation,
        )
        raw = None
        payload = default_session_payload(generation)
    elif not raw:
        payload = default_session_payload(generation)
    if _payload_is_invalid(raw):
        logger.warning("会话 JSON 损坏，已重置: %s", session_id)
    if payload.get("messages") or str(payload.get("summary") or ""):
        return payload, client
    if not settings.postgres_restore_on_redis_miss:
        return payload, client
    snapshot = postgres_persistence.load_session_snapshot(
        session_id=session_id,
        turn_limit=max(1, int(settings.postgres_restore_turn_limit)),
        settings=settings,
    )
    restored_messages = list(snapshot.get("messages") or [])
    if not restored_messages and not str(snapshot.get("summary") or ""):
        return payload, client
    restored_payload = {
        "messages": restored_messages,
        "summary": str(snapshot.get("summary") or ""),
        "version": 0,
        "generation": generation,
    }
    payload, restored = _restore_session_if_unchanged(
        client,
        session_id,
        restored_payload,
        expected_raw=raw,
        ttl_sec=max(1, int(settings.postgres_restore_redis_ttl_sec)),
    )
    if restored:
        logger.info(
            "会话已从 PostgreSQL 恢复: session_id=%s turns=%s",
            session_id,
            len(restored_messages) // 2,
        )
    return payload, client


def append_turn_and_save(
    client: Any,
    session_id: str,
    payload: Dict[str, Any],
    user_text: str,
    assistant_text: str,
    *,
    turn_id: str = "",
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    stable_turn_id = str(turn_id or "")
    expected_generation = int(payload.get("generation") or 0)

    def mutate(current: Dict[str, Any]) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = list(current.get("messages") or [])
        if stable_turn_id and any(
            str(row.get("turn_id") or "") == stable_turn_id for row in rows
        ):
            return current
        now = int(time.time())
        rows.append(
            {
                "role": "user",
                "content": redact_memory_text(
                    user_text, enabled=settings.memory_redact_pii
                ),
                "turn_id": stable_turn_id,
                "created_at": now,
            }
        )
        rows.append(
            {
                "role": "assistant",
                "content": redact_memory_text(
                    assistant_text, enabled=settings.memory_redact_pii
                ),
                "turn_id": stable_turn_id,
                "created_at": now,
            }
        )
        current["messages"] = rows[-max(2, int(settings.memory_max_messages)) :]
        current["version"] = int(current.get("version") or 0) + 1
        return current

    updated = atomic_update_session(
        client,
        session_id,
        mutate,
        ttl_sec=settings.memory_ttl_sec,
        expected_generation=expected_generation,
    )
    payload.clear()
    payload.update(updated)
    return payload


def update_summary(
    client: Any,
    session_id: str,
    payload: Dict[str, Any],
    summary: str,
    *,
    settings: Optional[Settings] = None,
) -> bool:
    settings = settings or get_settings()
    expected_generation = int(payload.get("generation") or 0)
    expected_version = int(payload.get("version") or 0)

    def mutate(current: Dict[str, Any]) -> Dict[str, Any]:
        if int(current.get("version") or 0) != expected_version:
            raise SessionVersionMismatchError(
                "session changed while its summary was being generated"
            )
        current["summary"] = redact_memory_text(
            summary, enabled=settings.memory_redact_pii
        )
        current["summary_version"] = int(current.get("summary_version") or 0) + 1
        current["version"] = int(current.get("version") or 0) + 1
        return current

    try:
        updated = atomic_update_session(
            client,
            session_id,
            mutate,
            ttl_sec=settings.memory_ttl_sec,
            expected_generation=expected_generation,
        )
    except (
        SessionTombstonedError,
        SessionGenerationMismatchError,
        SessionVersionMismatchError,
    ):
        logger.info("会话已删除、换代或追加新消息，丢弃迟到摘要: session_id=%s", session_id)
        return False
    payload.clear()
    payload.update(updated)
    return True


def export_session_payload(
    client: Any,
    session_id: str,
    *,
    settings: Optional[Settings] = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    state = postgres_persistence.get_session_state(session_id, settings=settings)
    if state is None:
        raise RuntimeError("persistent memory tombstone state is unavailable")
    generation = max(
        int(state.get("generation") or 0),
        get_redis_session_generation(client, session_id),
    )
    if bool(state.get("tombstoned")) or is_session_tombstoned(client, session_id):
        return default_session_payload(generation)
    payload = load_session_payload(client, session_id)
    if int(payload.get("generation") or 0) != generation:
        return default_session_payload(generation)
    return payload


def delete_session_payload(
    client: Any,
    session_id: str,
    *,
    generation: Optional[int] = None,
) -> bool:
    return mark_session_deleted(client, session_id, generation=generation)


def mark_session_deleted(
    client: Any,
    session_id: str,
    *,
    tombstone_ttl_sec: int = DEFAULT_TOMBSTONE_TTL_SEC,
    generation: Optional[int] = None,
) -> bool:
    key = _key(session_id)
    tombstone_key = _tombstone_key(session_id)
    epoch_key = _epoch_key(session_id)
    ttl_sec = max(1, int(tombstone_ttl_sec))
    requested_generation = (
        max(0, int(generation)) if generation is not None else None
    )

    if isinstance(client, redis.Redis):
        script = (
            "local existed = redis.call('exists', KEYS[1]); "
            "local current = tonumber(redis.call('get', KEYS[3]) or '0'); "
            "local requested = tonumber(ARGV[1]); "
            "local next_generation = requested; "
            "if requested < 0 then next_generation = current + 1; end; "
            "if next_generation < current then next_generation = current; end; "
            "redis.call('del', KEYS[1]); "
            "redis.call('set', KEYS[2], tostring(next_generation), 'EX', ARGV[2]); "
            "redis.call('set', KEYS[3], tostring(next_generation)); "
            "return existed"
        )
        return bool(
            client.eval(
                script,
                3,
                key,
                tombstone_key,
                epoch_key,
                requested_generation if requested_generation is not None else -1,
                ttl_sec,
            )
        )

    with _lock_for_session(session_id):
        existed = bool(client.delete(key))
        current_generation = get_redis_session_generation(client, session_id)
        next_generation = (
            current_generation + 1
            if requested_generation is None
            else max(current_generation, requested_generation)
        )
        client.set(epoch_key, str(next_generation))
        client.set(tombstone_key, str(next_generation), ex=ttl_sec)
        return existed


def session_generation_is_current(
    client: Any,
    session_id: str,
    generation: int,
    *,
    settings: Optional[Settings] = None,
) -> bool:
    settings = settings or get_settings()
    expected = max(0, int(generation))
    if is_session_tombstoned(client, session_id):
        return False
    if get_redis_session_generation(client, session_id) != expected:
        return False
    state = postgres_persistence.get_session_state(session_id, settings=settings)
    if state is None:
        return False
    if not bool(state.get("persistent")):
        return True
    return not bool(state.get("tombstoned")) and int(
        state.get("generation") or 0
    ) == expected
