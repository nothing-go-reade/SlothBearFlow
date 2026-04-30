from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool


def build_get_session_context_tool(chat_history: List[BaseMessage]):

    @tool
    def get_session_context() -> str:
        if not chat_history:
            return "当前会话中还没有历史上下文。"

        lines: List[str] = []
        for idx, message in enumerate(chat_history[-6:], start=1):
            role = "user"
            if isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, HumanMessage):
                role = "user"
            text = " ".join(str(message.content).strip().split())
            lines.append(f"{idx}. {role}: {text[:160]}")
        return "最近会话上下文：\n" + "\n".join(lines)

    return get_session_context
