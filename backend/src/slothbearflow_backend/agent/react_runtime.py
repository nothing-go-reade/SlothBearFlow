from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    call_id: str


@dataclass
class ToolResultRecord:
    call_id: str
    name: str
    ok: bool
    content: str


@dataclass
class ReActRuntimeState:
    steps: int = 0
    stop_reason: str = ""
    tools_used: list[str] = field(default_factory=list)


class ExplicitReActRuntime:
    def __init__(
        self,
        *,
        llm: Any,
        tools: Iterable[Any],
        max_steps: int = 4,
    ) -> None:
        self._llm = llm
        self._tools = list(tools)
        self._tool_map = {tool.name: tool for tool in self._tools}
        self._max_steps = max(1, int(max_steps))

    def _build_messages(self, payload: dict[str, Any]) -> list[BaseMessage]:
        messages: list[BaseMessage] = []
        for m in payload.get("chat_history") or []:
            if isinstance(m, BaseMessage):
                messages.append(m)
        query = str(payload.get("input") or "").strip()
        if query:
            messages.append(HumanMessage(content=query))
        return messages

    def _extract_text(self, ai_msg: AIMessage) -> str:
        content = ai_msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part if isinstance(part, str) else str(part) for part in content)
        return str(content or "")

    def _normalize_tool_call(self, raw: dict[str, Any], index: int) -> ToolCallRecord:
        return ToolCallRecord(
            name=str(raw.get("name") or ""),
            args=dict(raw.get("args") or {}),
            call_id=str(raw.get("id") or f"tool_call_{index}"),
        )

    def _invoke_tool(self, call: ToolCallRecord) -> ToolResultRecord:
        tool = self._tool_map.get(call.name)
        if tool is None:
            return ToolResultRecord(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"Tool `{call.name}` is not available.",
            )
        try:
            result = tool.invoke(call.args if call.args else {})
            return ToolResultRecord(
                call_id=call.call_id,
                name=call.name,
                ok=True,
                content=str(result),
            )
        except Exception as exc:
            return ToolResultRecord(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=f"Tool `{call.name}` failed: {exc}",
            )

    def _run(self, payload: dict[str, Any]) -> tuple[str, ReActRuntimeState]:
        state = ReActRuntimeState()
        messages = self._build_messages(payload)
        system_prompt = payload.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.insert(0, SystemMessage(content=system_prompt))

        llm_with_tools = self._llm.bind_tools(self._tools)

        for step in range(1, self._max_steps + 1):
            state.steps = step
            ai_msg = llm_with_tools.invoke(messages)
            if not isinstance(ai_msg, AIMessage):
                ai_msg = AIMessage(content=str(ai_msg))
            messages.append(ai_msg)

            tool_calls = list(ai_msg.tool_calls or [])
            if not tool_calls:
                state.stop_reason = "final_answer"
                return self._extract_text(ai_msg).strip(), state

            for idx, raw_call in enumerate(tool_calls, start=1):
                call = self._normalize_tool_call(raw_call, idx)
                result = self._invoke_tool(call)
                if call.name and call.name not in state.tools_used:
                    state.tools_used.append(call.name)
                observation = {
                    "ok": result.ok,
                    "tool": result.name,
                    "content": result.content,
                }
                messages.append(
                    ToolMessage(
                        content=json.dumps(observation, ensure_ascii=False),
                        tool_call_id=result.call_id,
                    )
                )

        state.stop_reason = "max_steps"
        return ("I completed the reasoning steps but could not reach a stable final answer. "
                "Please provide more detail and I will continue."), state

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        output, state = self._run(payload)
        return {
            "output": output,
            "stop_reason": state.stop_reason,
            "steps": state.steps,
            "tools_used": state.tools_used,
        }

    def stream(self, payload: dict[str, Any]):
        result = self.invoke(payload)
        text = str(result.get("output") or "")
        if text:
            yield {"output": text}
