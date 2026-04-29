from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage


def trim_message_window(
    messages: List[BaseMessage], max_pairs: int
) -> List[BaseMessage]:
    """保留最近 max_pairs 轮对话（按 user+assistant 估算，每轮 2 条消息）。"""
    if max_pairs <= 0:
        return []
    limit = max_pairs * 2
    return messages[-limit:]
