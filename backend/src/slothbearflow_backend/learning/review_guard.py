from __future__ import annotations

import threading
from typing import Optional, Set

_local = threading.local()

_DEFAULT_DENY = (
    "Background review denied non-whitelisted tool: {tool_name}. "
    "Only memory/skill tools are allowed."
)


def set_thread_tool_whitelist(
    names: Set[str], deny_msg_fmt: str = _DEFAULT_DENY
) -> None:
    """安装线程级工具白名单（对标 Hermes set_thread_tool_whitelist）。

    仅作用于当前线程；复盘 job 跑在 worker 线程，故只约束复盘。
    """
    _local.whitelist = set(names)
    _local.deny_fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _local.whitelist = None
    _local.deny_fmt = _DEFAULT_DENY


def get_thread_tool_whitelist() -> Optional[Set[str]]:
    return getattr(_local, "whitelist", None)


def is_tool_allowed(name: str) -> bool:
    """未设置白名单（普通主链路）→ 全部放行；设置后只放行名单内工具。"""
    whitelist = get_thread_tool_whitelist()
    if whitelist is None:
        return True
    return name in whitelist


def deny_message(name: str) -> str:
    fmt = getattr(_local, "deny_fmt", _DEFAULT_DENY)
    try:
        return fmt.format(tool_name=name)
    except Exception:
        return _DEFAULT_DENY.format(tool_name=name)
