from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple


class RateLimitExceeded(RuntimeError):
    pass


class RateLimitUnavailable(RuntimeError):
    pass


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, *, limit: int, window_sec: float = 60.0) -> None:
        now = time.monotonic()
        cutoff = now - window_sec
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                raise RateLimitExceeded("rate limit exceeded")
            events.append(now)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


rate_limiter = SlidingWindowRateLimiter()


def check_distributed_rate_limit(key: str, *, limit: int, settings: object) -> None:
    from backend.src.slothbearflow_backend.deps import get_redis

    production = str(getattr(settings, "app_env", "local")) in {
        "staging",
        "production",
    }
    try:
        client = get_redis(settings, allow_fallback=not production)
    except Exception as exc:
        raise RateLimitUnavailable("distributed rate limiter is unavailable") from exc
    bucket = int(time.time() // 60)
    redis_key = f"rate:{bucket}:{key}"
    try:
        pipeline_factory = getattr(client, "pipeline", None)
        if callable(pipeline_factory):
            with client.pipeline() as pipe:
                pipe.incr(redis_key)
                pipe.expire(redis_key, 90)
                count, _ = pipe.execute()
        else:
            count = client.incr(redis_key)
            client.expire(redis_key, 90)
        if int(count) > int(limit):
            raise RateLimitExceeded("rate limit exceeded")
    except RateLimitExceeded:
        raise
    except Exception as exc:
        if production:
            raise RateLimitUnavailable(
                "distributed rate limiter is unavailable"
            ) from exc
        # The process-local limiter remains a deterministic fallback if Redis
        # is degraded in local development. Production fails closed above.
        rate_limiter.check(key, limit=limit)
