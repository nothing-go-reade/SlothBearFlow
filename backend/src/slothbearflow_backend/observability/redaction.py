from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


REDACTED = "[REDACTED]"

_CONTENT_FIELDS = frozenset(
    {
        "answer",
        "body",
        "citation",
        "citations",
        "content",
        "context",
        "document",
        "documents",
        "excerpt",
        "input",
        "inputs",
        "input_summary",
        "message",
        "messages",
        "observation",
        "output",
        "outputs",
        "output_summary",
        "payload",
        "prompt",
        "prompts",
        "provenance",
        "query",
        "raw",
        "raw_output",
        "request_body",
        "response_body",
        "text",
    }
)
_CONTENT_SUFFIXES = (
    "_answer",
    "_body",
    "_citation",
    "_citations",
    "_content",
    "_context",
    "_excerpt",
    "_input",
    "_inputs",
    "_message",
    "_messages",
    "_observation",
    "_output",
    "_outputs",
    "_payload",
    "_prompt",
    "_prompts",
    "_provenance",
    "_query",
    "_text",
)
_SECRET_FIELDS = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "dsn",
        "access_token",
        "client_secret",
        "id_token",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "token",
    }
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_password",
    "_private_key",
    "_secret",
    "_secret_key",
    "_token",
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/-]{8,}={0,2}"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
        r"-----END [A-Z ]*PRIVATE KEY-----"
    ),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|accesstoken|refresh[_-]?token|"
        r"refreshtoken|client[_-]?secret|clientsecret|private[_-]?key|privatekey|"
        r"token|password|secret)\s*[:=]\s*[^\s'\"]{6,}"
    ),
)


def sanitize_observability_data(value: Any, *, include_content: bool) -> Any:
    """Return a detached trace-safe value for both local storage and exporters."""

    return _sanitize(value, include_content=include_content, redact_content=False)


def _sanitize(value: Any, *, include_content: bool, redact_content: bool) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if redact_content:
            return REDACTED if value else ""
        return _redact_secrets(value)
    if isinstance(value, bytes):
        if redact_content:
            return REDACTED if value else ""
        return _redact_secrets(value.decode("utf-8", errors="replace"))

    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            normalized_key = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key).strip()
            ).lower().replace("-", "_")
            secret_field = _is_secret_field(normalized_key)
            content_field = not include_content and _is_content_field(normalized_key)
            sanitized[key] = _sanitize(
                item,
                include_content=include_content,
                redact_content=redact_content or secret_field or content_field,
            )
        return sanitized
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _sanitize(
                item,
                include_content=include_content,
                redact_content=redact_content,
            )
            for item in value
        ]

    if redact_content:
        return REDACTED
    return _redact_secrets(str(value))


def _is_content_field(key: str) -> bool:
    return key in _CONTENT_FIELDS or key.endswith(_CONTENT_SUFFIXES)


def _is_secret_field(key: str) -> bool:
    return key in _SECRET_FIELDS or key.endswith(_SECRET_SUFFIXES)


def _redact_secrets(value: str) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result
