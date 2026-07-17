from __future__ import annotations

import contextvars
import copy
import json
from typing import Any, Dict, List, Optional

from backend.src.slothbearflow_backend.security.scrub import scrub_observation


_tool_trace: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar("agent_tool_trace", default=None)
)


def begin_tool_trace() -> None:
    _tool_trace.set([])


def end_tool_trace() -> None:
    _tool_trace.set(None)


def record_tool_trace(item: Dict[str, Any]) -> None:
    trace = _tool_trace.get()
    if trace is not None:
        trace.append(copy.deepcopy(item))


def get_tool_trace() -> List[Dict[str, Any]]:
    return copy.deepcopy(_tool_trace.get() or [])


def safe_tool_args(args: Dict[str, Any], settings: Any) -> Dict[str, Any]:
    raw = json.dumps(args, ensure_ascii=False, default=str)
    scrubbed = str(scrub_observation(raw, settings))
    try:
        value = json.loads(scrubbed)
        return value if isinstance(value, dict) else {"value": value}
    except json.JSONDecodeError:
        return {"summary": scrubbed[:500]}
