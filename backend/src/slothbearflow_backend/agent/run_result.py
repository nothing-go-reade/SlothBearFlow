from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from backend.src.slothbearflow_backend.agent.content import extract_model_text


VALID_STOP_REASONS = {
    "final_answer",
    "max_steps",
    "max_tool_calls",
    "max_execution_time",
    "tool_timeout",
    "tool_denied",
    "repeated_tool_call",
    "cancelled",
    "model_error",
}


@dataclass
class ToolTraceItem:
    call_id: str
    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    ok: bool = False
    status: str = "failed"
    duration_ms: float = 0.0
    observation: str = ""
    error_code: str = ""
    policy_decision: str = "allow"
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRunResult:
    output: str
    stop_reason: str = "final_answer"
    steps: int = 0
    tools_used: List[str] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    rag_sources: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""
    executor: str = "basic"
    prompt_version: str = "v1"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["rag_citations"] = list(value["citations"])
        return value

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        latency_ms: float = 0.0,
        model: str = "",
        executor: str = "basic",
        prompt_version: str = "v1",
    ) -> "AgentRunResult":
        data = payload if isinstance(payload, dict) else {"output": payload}
        stop_reason = str(data.get("stop_reason") or "final_answer")
        if stop_reason not in VALID_STOP_REASONS:
            stop_reason = "model_error"
        trace = [dict(item) for item in (data.get("tool_trace") or [])]
        tools = _unique_strings(
            list(data.get("tools_used") or [])
            + [str(item.get("name") or "") for item in trace]
        )
        return cls(
            output=extract_model_text(data.get("output")),
            stop_reason=stop_reason,
            steps=max(0, int(data.get("steps") or len(trace))),
            tools_used=tools,
            tool_trace=trace,
            citations=[
                dict(item)
                for item in (
                    data.get("rag_citations") or data.get("citations") or []
                )
            ],
            rag_sources=_unique_strings(data.get("rag_sources") or []),
            latency_ms=round(float(data.get("latency_ms") or latency_ms), 3),
            model=str(data.get("model") or model),
            executor=str(data.get("executor") or executor),
            prompt_version=str(data.get("prompt_version") or prompt_version),
        )


def _unique_strings(values: Iterable[Any]) -> List[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value or "").strip()))
