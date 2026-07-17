from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List


_TEXT_BLOCK_TYPES = {"text", "output_text"}


def extract_model_text(value: Any) -> str:
    """Extract user-visible text without stringifying reasoning/tool blocks."""

    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for block in content:
            text = _text_from_block(block)
            if text:
                parts.append(text)
        return "".join(parts)
    return _text_from_block(content)


def _text_from_block(block: Any) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        block_type = str(block.get("type") or "").strip().lower()
        if block_type not in _TEXT_BLOCK_TYPES:
            return ""
        for key in ("text", "output_text", "content"):
            value = block.get(key)
            if isinstance(value, str):
                return value
        return ""
    block_type = str(getattr(block, "type", "") or "").strip().lower()
    if block_type not in _TEXT_BLOCK_TYPES:
        return ""
    for attribute in ("text", "output_text", "content"):
        value = getattr(block, attribute, None)
        if isinstance(value, str):
            return value
    return ""
