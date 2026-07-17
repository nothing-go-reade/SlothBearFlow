from __future__ import annotations

import logging
import fnmatch
import time
from typing import Any, Optional

import threading

import redis

from backend.src.slothbearflow_backend import Settings, get_settings

_lock = threading.Lock()

logger = logging.getLogger(__name__)

_redis_client: Optional[Any] = None
_fallback_client: Optional[Any] = None
_last_redis_attempt = 0.0
_last_redis_healthcheck = 0.0


class InMemoryRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._expires: dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            expires_at = self._expires.get(key)
            if expires_at is not None and time.monotonic() >= expires_at:
                self._store.pop(key, None)
                self._expires.pop(key, None)
                return None
            return self._store.get(key)

    def set(
        self,
        key: str,
        value: str,
        ex: Optional[int] = None,
        nx: bool = False,
    ) -> bool:
        with self._lock:
            if nx and self.get(key) is not None:
                return False
            self._store[key] = value
            if ex is not None:
                self._expires[key] = time.monotonic() + max(1, int(ex))
            else:
                self._expires.pop(key, None)
        return True

    def delete(self, key: str) -> int:
        with self._lock:
            existed = key in self._store
            self._store.pop(key, None)
            self._expires.pop(key, None)
            return int(existed)

    def incr(self, key: str) -> int:
        with self._lock:
            current = int(self.get(key) or 0) + 1
            self._store[key] = str(current)
            return current

    def expire(self, key: str, seconds: int) -> bool:
        with self._lock:
            if key not in self._store:
                return False
            self._expires[key] = time.monotonic() + max(1, int(seconds))
            return True

    def scan_iter(self, match: str = "*"):
        with self._lock:
            keys = list(self._store)
        for key in keys:
            if fnmatch.fnmatch(key, match) and self.get(key) is not None:
                yield key

    def ping(self) -> bool:
        return True


def get_redis(
        settings: Optional[Settings] = None,
        *,
        allow_fallback: bool = True,
        force_probe: bool = False,
) -> Any:
    global _redis_client, _fallback_client, _last_redis_attempt, _last_redis_healthcheck

    if _redis_client is not None:
        settings = settings or get_settings()
        interval = max(0.1, float(settings.redis_retry_interval_sec))
        if not force_probe and time.monotonic() - _last_redis_healthcheck < interval:
            return _redis_client
        try:
            _redis_client.ping()
            _last_redis_healthcheck = time.monotonic()
            return _redis_client
        except redis.RedisError:
            with _lock:
                _redis_client = None

    with _lock:
        if _redis_client is not None:
            return _redis_client

        settings = settings or get_settings()
        now = time.monotonic()
        retry_interval = max(0.1, float(settings.redis_retry_interval_sec))
        if allow_fallback and _fallback_client is not None and now - _last_redis_attempt < retry_interval:
            return _fallback_client

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
        _last_redis_attempt = now

        try:
            client.ping()
            _redis_client = client
            _last_redis_healthcheck = time.monotonic()
        except redis.RedisError as exc:
            if not allow_fallback:
                raise
            logger.warning("Redis 不可用，回退到内存会话存储: %s", exc)
            if _fallback_client is None:
                _fallback_client = InMemoryRedis()
            return _fallback_client

        return _redis_client


def ping_redis(settings: Optional[Settings] = None) -> tuple[bool, Optional[str]]:
    try:
        r = get_redis(settings, allow_fallback=False, force_probe=True)
        if r.ping():
            return True, None
        return False, "PING failed"
    except redis.RedisError as e:
        logger.debug("Redis ping error: %s", e)
        return False, str(e)


def reset_redis_clients() -> None:
    global _redis_client, _fallback_client, _last_redis_attempt, _last_redis_healthcheck
    with _lock:
        _redis_client = None
        _fallback_client = None
        _last_redis_attempt = 0.0
        _last_redis_healthcheck = 0.0
