from __future__ import annotations

from typing import Any, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import RunnableSerializable

from backend.src.slothbearflow_backend import (
    Settings,
    build_tools,
    get_agent_prompt,
    get_chat_llm,
    get_settings,
    llm_supports_tools,
)
from backend.src.slothbearflow_backend.agent.react_runtime import ExplicitReActRuntime
from backend.src.slothbearflow_backend.prompt import build_system_prompt, get_basic_chat_prompt


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
) -> Any:
    settings = settings or get_settings()
    llm = get_chat_llm(settings)

    if not llm_supports_tools(settings):
        #  普通提示词: Prompt = System + History + Input
        prompt = get_basic_chat_prompt(
            rolling_summary=rolling_summary,
            structured_output=settings.structured_output,
        )
        return BasicChatExecutor(prompt | llm)

    # 调试——工具
    tools = build_tools(vector_store, chat_history=chat_history, settings=settings)

    # ReAct Agent 追踪
    if settings.enable_explicit_react_runtime:
        system_prompt = build_system_prompt(
            rolling_summary=rolling_summary,
            supports_tools=True,
            structured_output=settings.structured_output,
        )

        class ReActExecutorAdapter(ExplicitReActRuntime):
            def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
                runtime_payload = {
                    **payload,
                    "chat_history": list(chat_history or []),
                    "system_prompt": system_prompt,
                }
                return super().invoke(runtime_payload)

            def stream(self, payload: dict[str, Any]):
                runtime_payload = {
                    **payload,
                    "chat_history": list(chat_history or []),
                    "system_prompt": system_prompt,
                }
                yield from super().stream(runtime_payload)

        return ReActExecutorAdapter(
            llm=llm,
            tools=tools,
            max_steps=settings.react_max_steps,
        )

    # 目前单轮 标准ReAct  Agent提示词：Prompt = System + History + User + Scratchpad
    # TODO 思考1： 没有限制 思考长度 + 调用次数 + 工具选择策略 （问题 无限思考 + 循环调用工具）
    #         思考2： Agent 的中间过程 Thought / Action / Observation， 但是structured_output限制为JSON 会导致乱输出JSON 或者工具调用失败
    #            思考3： 完整提示词 Prompt = System + Summary + Selected History + Retrieved Context + Tools + Input ？
    #            改造1：强化工具调用规则    已做
    #            改造2：限制 Agent 行为    未做 （建议最多调用工具 3 次 每次必须基于 observation 决策）
    #            改造3：明确最终输出阶段 （当你完成所有工具调用后，必须生成最终答案  不得继续输出 Thought / Action）
    #            改造4：RAG 要“结构化注入”，不是靠 Prompt  结构占位提示词
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
