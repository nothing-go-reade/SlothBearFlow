from __future__ import annotations

from typing import Any, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableSerializable

from slothbearflow_backend import (
    Settings,
    build_tools,
    get_agent_prompt,
    get_chat_llm,
    get_settings,
    llm_supports_tools,
)
from slothbearflow_backend.prompt import get_basic_chat_prompt


class BasicChatExecutor:
    def __init__(self, runnable: RunnableSerializable[Any, Any]) -> None:
        self._runnable = runnable

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._runnable.invoke(payload)
        return {"output": str(result.content if hasattr(result, "content") else result)}

    def stream(self, payload: dict[str, Any]):
        for chunk in self._runnable.stream(payload):
            content = getattr(chunk, "content", chunk)
            if isinstance(content, list):
                text = "".join(
                    part if isinstance(part, str) else str(part) for part in content
                )
            else:
                text = str(content)
            if text:
                yield {"output": text}


def build_agent_executor(
    *,
    vector_store: Optional[Any],
    chat_history: Optional[list[Any]] = None,
    rolling_summary: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> AgentExecutor:
    """构建生产级 AgentExecutor（导入路径已全局修复为 langchains.app）。"""
    settings = settings or get_settings()
    llm = get_chat_llm(settings)

    if not llm_supports_tools(settings):
        prompt = get_basic_chat_prompt(
            rolling_summary=rolling_summary,
            structured_output=settings.structured_output,
        )
        return BasicChatExecutor(prompt | llm)

    tools = build_tools(vector_store, chat_history=chat_history, settings=settings)
    prompt = get_agent_prompt(
        rolling_summary=rolling_summary,
        structured_output=settings.structured_output,
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        early_stopping_method="generate",
    )
