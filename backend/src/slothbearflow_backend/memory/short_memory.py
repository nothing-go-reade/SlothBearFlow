from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.messages import BaseMessage


def trim_message_window(
    messages: List[BaseMessage], max_pairs: int, max_tokens: int = 0
) -> List[BaseMessage]:
    if max_pairs <= 0:
        return []
    limit = max_pairs * 2
    selected = messages[-limit:]
    if max_tokens <= 0:
        return selected
    output: List[BaseMessage] = []
    used = 0
    for message in reversed(selected):
        cost = estimate_tokens(getattr(message, "content", "")) + 4
        if output and used + cost > max_tokens:
            break
        if not output and cost > max_tokens:
            available = max(0, max_tokens - 4)
            truncated = _truncate_message(message, available)
            if truncated is not None:
                output.append(truncated)
            break
        output.append(message)
        used += cost
    output.reverse()
    return output


def _truncate_message(message: BaseMessage, max_tokens: int) -> Optional[BaseMessage]:
    if max_tokens <= 0:
        return None
    content = str(getattr(message, "content", "") or "")
    if not content:
        return message
    low, high = 0, len(content)
    while low < high:
        midpoint = (low + high + 1) // 2
        if estimate_tokens(content[:midpoint]) <= max_tokens:
            low = midpoint
        else:
            high = midpoint - 1
    if low <= 0:
        return None
    truncated = content[:low].rstrip()
    if low < len(content):
        truncated = truncated.rstrip().rstrip(".") + "..."
        while truncated and estimate_tokens(truncated) > max_tokens:
            truncated = truncated[:-4].rstrip() + "..."
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": truncated})
    return type(message)(content=truncated)


def estimate_tokens(value: Any) -> int:
    text = str(value or "")
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        ascii_count = sum(1 for char in text if ord(char) < 128)
        non_ascii_count = len(text) - ascii_count
        return max(1, ascii_count // 4 + non_ascii_count)
