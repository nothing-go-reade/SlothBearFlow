from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TraceContext:
    trace_id: str
    request_id: str
    operation: str
    user_id: str = "anonymous"
    tenant_id: str = "local"
    error_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


_trace_context: contextvars.ContextVar[Optional[TraceContext]] = (
    contextvars.ContextVar("slothbearflow_trace_context", default=None)
)


def new_trace_context(
    operation: str,
    *,
    request_id: str = "",
    trace_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> TraceContext:
    return TraceContext(
        trace_id=trace_id or uuid.uuid4().hex,
        request_id=request_id or uuid.uuid4().hex,
        operation=operation,
        metadata=dict(metadata or {}),
    )


def bind_trace_context(context: TraceContext) -> contextvars.Token:
    return _trace_context.set(context)


def reset_trace_context(token: contextvars.Token) -> None:
    _trace_context.reset(token)


def current_trace_context() -> Optional[TraceContext]:
    return _trace_context.get()


def current_trace_id() -> str:
    context = current_trace_context()
    return context.trace_id if context else ""


def set_identity(*, user_id: str, tenant_id: str) -> None:
    context = current_trace_context()
    if context is not None:
        context.user_id = user_id
        context.tenant_id = tenant_id


def mark_trace_error(error_type: str) -> None:
    context = current_trace_context()
    if context is not None:
        context.error_type = str(error_type or "stream_error")[:128]
