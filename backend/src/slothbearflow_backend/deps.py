from __future__ import annotations

import logging
from typing import Any, Optional

import threading

import redis

from backend.src.slothbearflow_backend import Settings, get_settings

_lock = threading.Lock()

logger = logging.getLogger(__name__)

_redis_client: Optional[Any] = None


class InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._store[key] = value
        return True

    def ping(self) -> bool:
        return True


def get_redis(
        settings: Optional[Settings] = None, *, allow_fallback: bool = True
) -> Any:
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    with _lock:
        if _redis_client is not None:
            return _redis_client

        settings = settings or get_settings()

        kwargs: dict[str, Any] = {
            "host": settings.redis_host,
            "port": settings.redis_port,
            "db": settings.redis_db,
            "decode_responses": True,
            "socket_connect_timeout": settings.redis_socket_connect_timeout,
            "socket_timeout": settings.redis_socket_timeout,
        }

        if settings.redis_password:
            kwargs["password"] = settings.redis_password

        client = redis.Redis(**kwargs)

        try:
            client.ping()
            _redis_client = client
        except redis.RedisError as exc:
            if not allow_fallback:
                raise
            logger.warning("Redis 不可用，回退到内存会话存储: %s", exc)
            _redis_client = InMemoryRedis()

        return _redis_client


def ping_redis(settings: Optional[Settings] = None) -> tuple[bool, Optional[str]]:
    try:
        r = get_redis(settings, allow_fallback=False)
        if r.ping():
            return True, None
        return False, "PING failed"
    except redis.RedisError as e:
        logger.debug("Redis ping error: %s", e)
        return False, str(e)
