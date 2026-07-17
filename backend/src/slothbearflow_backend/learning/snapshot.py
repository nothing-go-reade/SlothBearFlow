from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class TurnSnapshot:
    """一轮对话的只读快照，交给后台 review agent 复盘。

    review 只消费该快照的副本，不持有任何活会话对象。
    """

    session_id: str
    user_message: str
    final_answer: str
    turn_id: str = ""
    generation: int = 0
    user_id: str = "local-user"
    tenant_id: str = "local"
    raw_output: str = ""
    tools_used: List[str] = field(default_factory=list)
    # 工具调用轨迹 [{name, args, observation}]；BasicChatExecutor / 关闭时为空。
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    rag_context: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    rolling_summary: str = ""
    # 本轮该复盘哪些维度（由 nudge 间隔决定）。
    review_memory: bool = False
    review_skills: bool = False

    def __post_init__(self) -> None:
        try:
            from backend.src.slothbearflow_backend.memory.redis_memory import (
                current_session_generation,
            )

            current = current_session_generation(self.session_id)
            if current is not None:
                self.generation = max(0, int(current))
        except Exception:  # noqa: BLE001
            self.generation = max(0, int(self.generation or 0))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnSnapshot":
        return cls(
            session_id=str(data.get("session_id") or ""),
            user_message=str(data.get("user_message") or ""),
            final_answer=str(data.get("final_answer") or ""),
            turn_id=str(data.get("turn_id") or ""),
            generation=max(0, int(data.get("generation") or 0)),
            user_id=str(data.get("user_id") or "local-user"),
            tenant_id=str(data.get("tenant_id") or "local"),
            raw_output=str(data.get("raw_output") or ""),
            tools_used=list(data.get("tools_used") or []),
            tool_trace=list(data.get("tool_trace") or []),
            rag_context=str(data.get("rag_context") or ""),
            citations=list(data.get("citations") or []),
            rolling_summary=str(data.get("rolling_summary") or ""),
            review_memory=bool(data.get("review_memory")),
            review_skills=bool(data.get("review_skills")),
        )

    def render(self, *, max_chars: int = 6000) -> str:
        """渲染为供 review agent 阅读的文本块。"""
        parts: List[str] = []
        if self.rolling_summary.strip():
            parts.append(f"【历史会话摘要】\n{self.rolling_summary.strip()}")
        parts.append(f"【用户问题】\n{self.user_message.strip()}")
        parts.append(f"【助手回答】\n{self.final_answer.strip()}")
        if self.tools_used:
            parts.append("【本轮使用工具】\n" + ", ".join(self.tools_used))
        if self.tool_trace:
            trace_lines: List[str] = []
            for step in self.tool_trace:
                name = str(step.get("name") or "")
                args = step.get("args")
                obs = str(step.get("observation") or "")
                trace_lines.append(
                    f"- {name} args={args} -> {obs[:240]}"
                )
            parts.append("【工具调用轨迹】\n" + "\n".join(trace_lines))
        if self.rag_context.strip():
            parts.append("【本轮检索片段】\n" + self.rag_context.strip()[:1200])
        block = "\n\n".join(parts)
        return block[:max_chars]
