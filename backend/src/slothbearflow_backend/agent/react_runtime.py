from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.src.slothbearflow_backend.agent.content import extract_model_text


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
    tool_trace: list = field(default_factory=list)
    call_fingerprints: set[str] = field(default_factory=set)


class ExplicitReActRuntime:
    def __init__(
        self,
        *,
        llm: Any,
        tools: Iterable[Any],
        max_steps: int = 4,
        max_tool_calls: Optional[int] = None,
    ) -> None:
        self._llm = llm
        self._tools = list(tools)
        self._tool_map = {tool.name: tool for tool in self._tools}
        self._max_steps = max(1, int(max_steps))
        self._max_tool_calls = max(
            1,
            int(max_tool_calls if max_tool_calls is not None else max_steps),
        )

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
        return extract_model_text(ai_msg)

    def _normalize_tool_call(self, raw: dict[str, Any], index: int) -> ToolCallRecord:
        return ToolCallRecord(
            name=str(raw.get("name") or ""),
            args=dict(raw.get("args") or {}),
            call_id=str(raw.get("id") or f"tool_call_{index}"),
        )

    def _invoke_tool(self, call: ToolCallRecord) -> ToolResultRecord:
        # 线程级白名单（仅后台复盘会设置）：放行名单外的工具直接 deny。
        from backend.src.slothbearflow_backend.learning.review_guard import (
            deny_message,
            is_tool_allowed,
        )

        if not is_tool_allowed(call.name):
            return ToolResultRecord(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=deny_message(call.name),
            )
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
                content=f"Tool `{call.name}` failed safely ({type(exc).__name__}).",
            )

    def _call_fingerprint(self, call: ToolCallRecord) -> str:
        return "%s:%s" % (
            call.name,
            json.dumps(call.args, sort_keys=True, ensure_ascii=False, default=str),
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
                if len(state.tool_trace) >= self._max_tool_calls:
                    state.stop_reason = "max_tool_calls"
                    return (
                        "I stopped because the tool-call limit for this turn was reached.",
                        state,
                    )
                call = self._normalize_tool_call(raw_call, idx)
                fingerprint = self._call_fingerprint(call)
                repeated = fingerprint in state.call_fingerprints
                state.call_fingerprints.add(fingerprint)
                if repeated:
                    result = ToolResultRecord(
                        call_id=call.call_id,
                        name=call.name,
                        ok=False,
                        content="Repeated tool call with unchanged arguments was blocked.",
                    )
                else:
                    result = self._invoke_tool(call)
                if call.name and call.name not in state.tools_used:
                    state.tools_used.append(call.name)
                state.tool_trace.append(
                    {
                        "name": result.name,
                        "args": call.args,
                        "ok": result.ok,
                        "observation": result.content,
                    }
                )
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
                if repeated:
                    state.stop_reason = "repeated_tool_call"
                    return (
                        "I stopped because the same tool call was repeated without new evidence.",
                        state,
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
            "tool_trace": state.tool_trace,
        }

    def stream(self, payload: dict[str, Any]):
        result = self.invoke(payload)
        text = str(result.get("output") or "")
        if text:
            yield {"output": text}
