from __future__ import annotations

from typing import List

from langchain_core.messages import BaseMessage


def trim_message_window(
    messages: List[BaseMessage], max_pairs: int
) -> List[BaseMessage]:
    if max_pairs <= 0:
        return []
    limit = max_pairs * 2
    return messages[-limit:]
