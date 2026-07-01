from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from backend.src.slothbearflow_backend.security.schema import PolicyBundle, ToolPolicy

logger = logging.getLogger(__name__)

# 内置默认策略里放行的当前只读工具（文件缺失 / PyYAML 不可用时的代码兜底）。
_DEFAULT_READONLY_TOOLS = (
    "get_current_time",
    "get_weather",
    "get_session_context",
    "search_knowledge",
)


def _builtin_default_policy() -> PolicyBundle:
    """代码兜底策略：放行 4 个已知只读工具、拒绝未知工具。

    仅在策略文件缺失 / 不可解析 / PyYAML 不可用时使用，
    保证策略文件意外丢失也不 breaks 现有工具，且未知工具仍被拒。
    """
    tools = {name: ToolPolicy(allow=True) for name in _DEFAULT_READONLY_TOOLS}
    return PolicyBundle(version=1, default_action="deny", tools=tools)


def _load_from_file(path: Path) -> Optional[PolicyBundle]:
    try:
        import yaml  # PyYAML
    except Exception:
        logger.warning("PyYAML 不可用，工具策略回退内置默认")
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            logger.warning("工具策略文件不是映射结构，回退内置默认: %s", path)
            return None
        return PolicyBundle(**data)
    except Exception:
        logger.exception("工具策略文件解析失败，回退内置默认: %s", path)
        return None


@lru_cache
def load_tool_policy(policy_file: str = "", mode: str = "enforce") -> PolicyBundle:
    """加载并缓存工具安全策略。

    缓存键 (policy_file, mode) 均为可哈希标量，便于测试 cache_clear()。
    空路径 / 文件不存在 / 解析失败 → 内置默认策略。
    （mode 仅参与缓存键，实际裁决在 engine 依据 settings.tool_guard_mode 进行。）
    """
    if policy_file:
        path = Path(policy_file)
        if path.is_file():
            bundle = _load_from_file(path)
            if bundle is not None:
                return bundle
        else:
            logger.warning("工具策略文件不存在，回退内置默认: %s", policy_file)
    return _builtin_default_policy()


def get_tool_policy(settings: Any) -> PolicyBundle:
    """按 settings 解析策略（从缓存取）。"""
    return load_tool_policy(
        getattr(settings, "tool_policy_file", "") or "",
        str(getattr(settings, "tool_guard_mode", "enforce") or "enforce"),
    )
