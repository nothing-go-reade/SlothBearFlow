from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Set


_INJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?(?:prior|previous|earlier)\s+(?:instructions?|rules?)",
        r"(?:forget|override)\s+(all\s+)?(?:prior|previous|earlier)\s+(?:instructions?|rules?)",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"you\s+are\s+now\s+",
        r"developer\s+message",
        r"system\s+message",
        r"忽略.{0,8}(之前|以上|先前).{0,8}(指令|要求|规则)",
        r"无视.{0,8}(上述|以上|之前|先前).{0,8}(指令|要求|规则)",
        r"(?:覆盖|绕过|忘记).{0,8}(上述|以上|之前|先前).{0,8}(指令|要求|规则)",
        r"泄露.{0,8}(系统提示词|系统消息|开发者消息)",
        r"从现在开始你是",
    )
)

_PROMPT_METADATA_FIELDS = ("source", "title", "filename", "section")
_ACTIVE_SOURCE_SCHEMES = ("javascript:", "data:", "vbscript:")
_SECRET_VALUE_QUERY_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:从未记录|未记录|未记).{0,48}(?:密钥|密码|令牌|token|api[_ -]?key)",
        r"(?:生产|真实|当前|内部|未记录).{0,12}(?:密钥|密码|令牌|token|api[_ -]?key).{0,8}(?:是什么|值|内容|明文|给我|显示)",
        r"(?:密钥|密码|令牌|token|api[_ -]?key).{0,8}(?:是什么|值|内容|明文|给我|显示).{0,12}(?:生产|真实|当前|内部|未记录)",
        r"(?:what\s+is|show|reveal|give\s+me).{0,40}(?:production|actual|current|internal).{0,20}(?:secret|password|token|api[_ -]?key)",
        r"(?:production|actual|current|internal).{0,20}(?:secret|password|token|api[_ -]?key).{0,20}(?:value|plaintext)",
    )
)
_RECALL_TTL_SEC = 600.0
_recall_lock = threading.RLock()
_recall_scopes: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class RagAccessContext:
    tenant_id: str = "local"
    user_id: str = "local"
    roles: Set[str] = field(default_factory=lambda: {"viewer"})
    allow_legacy: bool = True


def normalize_knowledge_acl(metadata: Mapping[str, Any]) -> dict[str, Any]:
    tenant_id = str(metadata.get("tenant_id") or "local").strip() or "local"
    owner_id = str(metadata.get("owner_id") or "").strip()
    visibility = str(metadata.get("visibility") or "tenant").strip().lower()
    if visibility not in {"private", "tenant", "public"}:
        raise ValueError("Knowledge visibility must be private, tenant, or public.")
    if visibility == "private" and not owner_id:
        owner_id = "local" if tenant_id == "local" else ""
    if visibility == "private" and not owner_id:
        raise ValueError("Private knowledge requires an owner_id.")
    allowed_roles = sorted(
        {
            str(role).strip()
            for role in _as_iterable(metadata.get("allowed_roles"))
            if str(role).strip()
        }
    )
    if visibility == "public" and allowed_roles:
        raise ValueError("Public knowledge cannot also declare allowed_roles.")
    return {
        "tenant_id": tenant_id,
        "owner_id": owner_id,
        "visibility": visibility,
        "allowed_roles": allowed_roles,
    }


def begin_citation_recall(turn_id: str) -> None:
    stable_turn_id = str(turn_id or "").strip()
    if not stable_turn_id:
        return
    now = time.monotonic()
    with _recall_lock:
        _prune_recall_scopes(now)
        _recall_scopes[stable_turn_id] = {"created_at": now, "keys": set()}


def clear_citation_recall(turn_id: str) -> None:
    with _recall_lock:
        _recall_scopes.pop(str(turn_id or ""), None)


def record_recalled_metadata(metadata: Mapping[str, Any]) -> None:
    turn_id = _current_turn_id()
    if not turn_id:
        return
    source = str(metadata.get("source") or "").strip()
    if not source:
        return
    chunk_id = str(metadata.get("chunk_id") or "").strip()
    with _recall_lock:
        scope = _recall_scopes.get(turn_id)
        if scope is not None:
            scope["keys"].add((source, chunk_id))


def citation_is_from_current_recall(source: str, chunk_id: str) -> Optional[bool]:
    turn_id = _current_turn_id()
    if not turn_id:
        return None
    with _recall_lock:
        scope = _recall_scopes.get(turn_id)
        if scope is None:
            return None
        keys = set(scope["keys"])
    source = str(source or "").strip()
    chunk_id = str(chunk_id or "").strip()
    if chunk_id:
        return (source, chunk_id) in keys
    return (source, "") in keys


def _current_turn_id() -> str:
    try:
        from backend.src.slothbearflow_backend.security.turn_state import (
            current_turn_id,
        )

        return str(current_turn_id() or "")
    except Exception:  # noqa: BLE001
        return ""


def _prune_recall_scopes(now: float) -> None:
    expired = [
        turn_id
        for turn_id, scope in _recall_scopes.items()
        if now - float(scope.get("created_at") or 0.0) > _RECALL_TTL_SEC
    ]
    for turn_id in expired:
        _recall_scopes.pop(turn_id, None)


def contains_prompt_injection(text: str) -> bool:
    candidate = unicodedata.normalize("NFKC", str(text or ""))
    candidate = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", candidate)
    return any(pattern.search(candidate) for pattern in _INJECTION_PATTERNS)


def query_requests_secret_value(text: str) -> bool:
    candidate = str(text or "")
    return any(pattern.search(candidate) for pattern in _SECRET_VALUE_QUERY_PATTERNS)


def citation_source_is_safe(source: Any) -> bool:
    candidate = str(source or "").strip()
    if not candidate or len(candidate) > 2048:
        return False
    if any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in candidate):
        return False
    lowered = candidate.lower()
    if lowered.startswith(_ACTIVE_SOURCE_SCHEMES):
        return False
    return not contains_prompt_injection(candidate)


def metadata_contains_prompt_injection(metadata: Mapping[str, Any]) -> bool:
    for field_name in _PROMPT_METADATA_FIELDS:
        value = metadata.get(field_name)
        if value not in (None, "") and contains_prompt_injection(str(value)):
            return True
    return False


def document_is_authorized(
    metadata: Mapping[str, Any],
    access: RagAccessContext,
) -> bool:
    tenant_id = str(metadata.get("tenant_id") or "")
    owner_id = str(metadata.get("owner_id") or "")
    visibility = str(metadata.get("visibility") or "").strip().lower()
    allowed_roles = {
        str(role) for role in _as_iterable(metadata.get("allowed_roles")) if str(role).strip()
    }

    authorized = False
    if not visibility:
        if not access.allow_legacy:
            return False
        if tenant_id and tenant_id != access.tenant_id:
            return False
        if owner_id and owner_id != access.user_id:
            return False
        if allowed_roles and not allowed_roles.intersection(access.roles):
            return False
        authorized = True
    elif visibility == "public":
        authorized = tenant_id == access.tenant_id and not allowed_roles
    elif tenant_id == access.tenant_id:
        owner_allowed = visibility != "private" or owner_id == access.user_id
        role_allowed = not allowed_roles or bool(allowed_roles.intersection(access.roles))
        authorized = owner_allowed and role_allowed and visibility in {"tenant", "private"}
    if authorized:
        record_recalled_metadata(metadata)
    return authorized


def build_milvus_acl_filters(access: RagAccessContext) -> List[str]:
    visibility = 'metadata["visibility"]'
    tenant_id = 'metadata["tenant_id"]'
    owner_id = 'metadata["owner_id"]'
    allowed_roles = 'metadata["allowed_roles"]'
    tenant_value = _milvus_literal(access.tenant_id)
    user_value = _milvus_literal(access.user_id)
    roles = sorted({str(role).strip() for role in access.roles if str(role).strip()})

    role_filters = [f"array_length({allowed_roles}) == 0"]
    if roles:
        role_filters.append(f"json_contains_any({allowed_roles}, {_milvus_array_literal(roles)})")
        role_filters.extend(f"{allowed_roles} == {_milvus_literal(role)}" for role in roles)

    filters = [
        f'{visibility} == "public" and {tenant_id} == {tenant_value} '
        f"and array_length({allowed_roles}) == 0"
    ]
    tenant_scope = f'{visibility} == "tenant" and {tenant_id} == {tenant_value}'
    private_scope = (
        f'{visibility} == "private" and {tenant_id} == {tenant_value} '
        f"and {owner_id} == {user_value}"
    )
    for scope_filter in (tenant_scope, private_scope):
        filters.extend(f"{scope_filter} and {role_filter}" for role_filter in role_filters)

    if access.allow_legacy:
        for legacy_visibility in (f"not exists {visibility}", f'{visibility} == ""'):
            filters.append(f"{legacy_visibility} and {tenant_id} == {tenant_value}")
            if access.tenant_id == "local":
                filters.extend(
                    (
                        f"{legacy_visibility} and not exists {tenant_id}",
                        f'{legacy_visibility} and {tenant_id} == ""',
                    )
                )
    return filters


def _as_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        return value
    if value in (None, ""):
        return ()
    return (value,)


def _milvus_literal(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def _milvus_array_literal(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)
