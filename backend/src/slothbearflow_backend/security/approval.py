from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import redis

from backend.src.slothbearflow_backend.security.audit import audit_event
from backend.src.slothbearflow_backend.security.identity import Principal, current_principal
from backend.src.slothbearflow_backend.security.scrub import scrub_observation


_ID_PREFIX = "tool:approval:id:"
_FINGERPRINT_PREFIX = "tool:approval:fingerprint:"
_CREATE_APPROVAL_SCRIPT = (
    "local existing = redis.call('get', KEYS[1]); "
    "if existing then return {0, existing}; end; "
    "redis.call('set', KEYS[2], ARGV[2], 'EX', ARGV[3]); "
    "redis.call('set', KEYS[1], ARGV[1], 'EX', ARGV[3]); "
    "return {1, ARGV[1]}"
)


class ApprovalStoreUnavailable(RuntimeError):
    pass


@dataclass
class ApprovalRequest:
    approval_id: str
    fingerprint: str
    tool_name: str
    args_summary: str
    user_id: str
    tenant_id: str
    status: str
    created_at: float
    expires_at: float
    decided_by: str = ""
    decided_at: float = 0.0


class ApprovalStore:
    def __init__(self) -> None:
        self._items: Dict[str, ApprovalRequest] = {}
        self._lock = threading.RLock()
        self._fallback_client: Any = None

    def authorize_or_request(
        self,
        *,
        tool_name: str,
        args: Dict[str, Any],
        settings: Any,
    ) -> Tuple[bool, str]:
        principal = current_principal()
        fingerprint = _fingerprint(tool_name, args, principal)
        now = time.time()
        ttl = max(30, int(getattr(settings, "tool_approval_ttl_sec", 900)))
        with self._lock:
            self._expire(now)
            client = self._client(settings)
            existing_id = client.get(_FINGERPRINT_PREFIX + fingerprint)
            if existing_id:
                existing_id = _redis_text(existing_id)
                item = self._load_winner_with_retry(existing_id, settings)
                if item is None:
                    return False, existing_id
                if item.expires_at > now:
                    if item.status == "approved" and self._consume(item, settings):
                        audit_event(
                            settings,
                            "tool.approval_consumed",
                            actor=principal.user_id,
                            tenant_id=principal.tenant_id,
                            target=tool_name,
                            metadata={"approval_id": item.approval_id},
                        )
                        return True, item.approval_id
                    if item.status == "pending":
                        return False, item.approval_id
                self._release_fingerprint(item, settings)
            approval_id = uuid.uuid4().hex
            summary = str(scrub_observation(json.dumps(args, ensure_ascii=False), settings))[:1000]
            item = ApprovalRequest(
                approval_id=approval_id,
                fingerprint=fingerprint,
                tool_name=tool_name,
                args_summary=summary,
                user_id=principal.user_id,
                tenant_id=principal.tenant_id,
                status="pending",
                created_at=now,
                expires_at=now + ttl,
            )
            claimed, winner_id = self._claim_with_record(item, settings)
            if not claimed:
                winner = self._load_winner_with_retry(winner_id, settings)
                return False, winner.approval_id if winner is not None else winner_id
        audit_event(
            settings,
            "tool.approval_requested",
            actor=principal.user_id,
            tenant_id=principal.tenant_id,
            target=tool_name,
            metadata={"approval_id": approval_id},
        )
        return False, approval_id

    def decide(
        self,
        approval_id: str,
        *,
        approve: bool,
        actor: Principal,
        settings: Any,
    ) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            self._expire(now)
            item = self._load(approval_id, settings)
            if item is None or item.tenant_id != actor.tenant_id:
                return None
            if item.status != "pending":
                return asdict(item)
            item.status = "approved" if approve else "rejected"
            item.decided_by = actor.user_id
            item.decided_at = now
            self._save(item, settings)
            result = asdict(item)
        audit_event(
            settings,
            "tool.approval_decided",
            actor=actor.user_id,
            tenant_id=actor.tenant_id,
            target=item.tool_name,
            outcome=item.status,
            metadata={"approval_id": approval_id},
        )
        return result

    def list(
        self,
        principal: Principal,
        *,
        limit: int = 100,
        settings: Any = None,
    ) -> List[Dict[str, Any]]:
        if settings is None:
            from backend.src.slothbearflow_backend.config import get_settings

            settings = get_settings()
        now = time.time()
        with self._lock:
            self._expire(now)
            client = self._client(settings)
            items = []
            for key in client.scan_iter(match=_ID_PREFIX + "*"):
                item = self._load(str(key).replace(_ID_PREFIX, "", 1), settings)
                if item is not None:
                    items.append(item)
            if not items:
                items = list(self._items.values())
            rows = [asdict(item) for item in items if item.tenant_id == principal.tenant_id]
        rows.sort(key=lambda item: item["created_at"], reverse=True)
        return rows[: max(1, min(limit, 200))]

    def reset(self) -> None:
        with self._lock:
            self._items.clear()
            self._fallback_client = None
            try:
                from backend.src.slothbearflow_backend.config import get_settings

                client = self._client(get_settings())
                keys = list(client.scan_iter(match="tool:approval:*"))
                for key in keys:
                    client.delete(key)
            except Exception:
                pass

    def _expire(self, now: float) -> None:
        for item in self._items.values():
            if item.status in {"pending", "approved"} and item.expires_at <= now:
                item.status = "expired"

    def _client(self, settings: Any) -> Any:
        from backend.src.slothbearflow_backend.deps import InMemoryRedis, get_redis

        required = (
            "redis_host",
            "redis_port",
            "redis_db",
            "redis_retry_interval_sec",
        )
        if not all(hasattr(settings, name) for name in required):
            if self._fallback_client is None:
                self._fallback_client = InMemoryRedis()
            return self._fallback_client
        production = str(getattr(settings, "app_env", "local")) in {
            "staging",
            "production",
        }
        try:
            return get_redis(settings, allow_fallback=not production)
        except redis.RedisError as exc:
            raise ApprovalStoreUnavailable(
                "distributed approval state is unavailable"
            ) from exc

    def _load(self, approval_id: str, settings: Any) -> Optional[ApprovalRequest]:
        if not approval_id:
            return None
        client = self._client(settings)
        raw = client.get(_ID_PREFIX + approval_id)
        if raw:
            try:
                item = ApprovalRequest(**json.loads(raw))
                self._items[approval_id] = item
                return item
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
        return self._items.get(approval_id)

    def _save(self, item: ApprovalRequest, settings: Any) -> None:
        ttl = max(1, int(item.expires_at - time.time()))
        client = self._client(settings)
        client.set(
            _ID_PREFIX + item.approval_id,
            json.dumps(asdict(item), ensure_ascii=False),
            ex=ttl,
        )
        self._items[item.approval_id] = item

    def _load_winner_with_retry(
        self,
        approval_id: str,
        settings: Any,
        *,
        attempts: int = 5,
    ) -> Optional[ApprovalRequest]:
        for attempt in range(max(1, attempts)):
            item = self._load(approval_id, settings)
            if item is not None:
                return item
            if attempt + 1 < attempts:
                time.sleep(0.005 * (2**attempt))
        return None

    def _claim_with_record(
        self,
        item: ApprovalRequest,
        settings: Any,
    ) -> Tuple[bool, str]:
        client = self._client(settings)
        fingerprint_key = _FINGERPRINT_PREFIX + item.fingerprint
        approval_key = _ID_PREFIX + item.approval_id
        ttl = max(1, int(item.expires_at - time.time()))
        payload = json.dumps(asdict(item), ensure_ascii=False)

        if isinstance(client, redis.Redis):
            try:
                result = client.eval(
                    _CREATE_APPROVAL_SCRIPT,
                    2,
                    fingerprint_key,
                    approval_key,
                    item.approval_id,
                    payload,
                    ttl,
                )
                claimed = bool(int(result[0]))
                winner_id = _redis_text(result[1])
                if claimed:
                    self._items[item.approval_id] = item
                return claimed, winner_id
            except redis.RedisError:
                return self._claim_with_transaction(
                    client,
                    item,
                    fingerprint_key=fingerprint_key,
                    approval_key=approval_key,
                    payload=payload,
                    ttl=ttl,
                )

        client_lock = getattr(client, "_lock", self._lock)
        with client_lock:
            existing_id = client.get(fingerprint_key)
            if existing_id:
                return False, _redis_text(existing_id)
            client.set(approval_key, payload, ex=ttl)
            client.set(fingerprint_key, item.approval_id, ex=ttl)
        self._items[item.approval_id] = item
        return True, item.approval_id

    def _claim_with_transaction(
        self,
        client: redis.Redis,
        item: ApprovalRequest,
        *,
        fingerprint_key: str,
        approval_key: str,
        payload: str,
        ttl: int,
        max_retries: int = 8,
    ) -> Tuple[bool, str]:
        for _ in range(max_retries):
            try:
                with client.pipeline() as pipe:
                    pipe.watch(fingerprint_key)
                    existing_id = pipe.get(fingerprint_key)
                    if existing_id:
                        pipe.unwatch()
                        return False, _redis_text(existing_id)
                    pipe.multi()
                    pipe.set(approval_key, payload, ex=ttl)
                    pipe.set(fingerprint_key, item.approval_id, ex=ttl)
                    pipe.execute()
                    self._items[item.approval_id] = item
                    return True, item.approval_id
            except redis.WatchError:
                continue
        winner_id = client.get(fingerprint_key)
        if winner_id:
            return False, _redis_text(winner_id)
        raise RuntimeError("concurrent approval claim retry limit exceeded")

    def _consume(self, item: ApprovalRequest, settings: Any) -> bool:
        client = self._client(settings)
        try:
            import redis

            if isinstance(client, redis.Redis):
                key = _ID_PREFIX + item.approval_id
                fingerprint_key = _FINGERPRINT_PREFIX + item.fingerprint
                for _ in range(5):
                    try:
                        with client.pipeline() as pipe:
                            pipe.watch(key, fingerprint_key)
                            raw = pipe.get(key)
                            current = ApprovalRequest(**json.loads(raw)) if raw else None
                            if current is None or current.status != "approved":
                                return False
                            current_fingerprint = str(pipe.get(fingerprint_key) or "")
                            current.status = "consumed"
                            ttl = max(1, int(current.expires_at - time.time()))
                            pipe.multi()
                            pipe.set(
                                key,
                                json.dumps(asdict(current), ensure_ascii=False),
                                ex=ttl,
                            )
                            if current_fingerprint == current.approval_id:
                                pipe.delete(fingerprint_key)
                            pipe.execute()
                            self._items[current.approval_id] = current
                            return True
                    except redis.WatchError:
                        continue
                return False
        except ImportError:
            pass
        if item.status != "approved":
            return False
        item.status = "consumed"
        self._save(item, settings)
        self._release_fingerprint(item, settings)
        return True

    def _release_fingerprint(self, item: ApprovalRequest, settings: Any) -> None:
        client = self._client(settings)
        key = _FINGERPRINT_PREFIX + item.fingerprint
        try:
            import redis
        except ImportError:
            redis = None
        if redis is not None and isinstance(client, redis.Redis):
            try:
                client.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) else return 0 end",
                    1,
                    key,
                    item.approval_id,
                )
                return
            except redis.RedisError:
                return
        if str(client.get(key) or "") == item.approval_id:
            client.delete(key)


def _redis_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _fingerprint(tool_name: str, args: Dict[str, Any], principal: Principal) -> str:
    raw = json.dumps(
        {
            "tool": tool_name,
            "args": args,
            "tenant": principal.tenant_id,
            "user": principal.user_id,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


approval_store = ApprovalStore()
