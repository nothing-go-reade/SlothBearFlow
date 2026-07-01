from __future__ import annotations

import contextvars
from typing import Dict, Optional, Tuple

# 每回合工具调用计数。用 contextvars 而非 threading.local：
# - 能穿透 asyncio.to_thread（conversation_loop 在线程里跑 executor.invoke）；
# - 每个请求协程持有独立 context → 绝不跨请求泄漏（worker 线程复用也安全）。
_turn_counts: contextvars.ContextVar[Optional[Dict[str, int]]] = contextvars.ContextVar(
    "tool_turn_counts", default=None
)


def begin_turn() -> None:
    """回合开始时重置计数（ChatTurnRunner.prepare 调用）。"""
    _turn_counts.set({})


def end_turn() -> None:
    """回合结束清理（ChatTurnRunner._finalize 的 finally 调用）。"""
    _turn_counts.set(None)


def current_counts() -> Optional[Dict[str, int]]:
    return _turn_counts.get()


def record_and_check(
    name: str,
    *,
    per_tool_limit: Optional[int] = None,
    global_limit: Optional[int] = None,
) -> Tuple[bool, str]:
    """记录一次调用并检查配额。返回 (ok, reason)。

    未开回合（counts is None，如后台复盘/单元直调）→ 不限制。
    先判断再累加，使「第 N 次」正好命中上限；被拒的调用不消耗配额。
    """
    counts = _turn_counts.get()
    if counts is None:
        return True, ""
    total = sum(counts.values())
    current = counts.get(name, 0)
    if global_limit is not None and total >= global_limit:
        return False, "tool call quota exceeded (max_tool_calls_per_turn=%s)" % global_limit
    if per_tool_limit is not None and current >= per_tool_limit:
        return False, "tool `%s` call quota exceeded (max_calls_per_turn=%s)" % (
            name,
            per_tool_limit,
        )
    counts[name] = current + 1
    return True, ""
