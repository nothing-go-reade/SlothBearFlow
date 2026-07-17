from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.src.slothbearflow_backend.security.schema import PolicyBundle, ToolPolicy

logger = logging.getLogger(__name__)

# 内置默认策略里放行的当前只读工具（文件缺失 / PyYAML 不可用时的代码兜底）。
_DEFAULT_READONLY_TOOLS = (
    "get_current_time",
    "get_weather",
    "get_session_context",
    "search_knowledge",
)


class PolicyLoadError(RuntimeError):
    """A policy could not be loaded and permissive fallback was forbidden."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"tool policy load failed ({reason}): {path}")


def _builtin_default_policy() -> PolicyBundle:
    """代码兜底策略：放行 4 个已知只读工具、拒绝未知工具。

    仅在策略文件缺失 / 不可解析 / PyYAML 不可用时使用，
    保证策略文件意外丢失也不 breaks 现有工具，且未知工具仍被拒。
    """
    tools = {name: ToolPolicy(allow=True) for name in _DEFAULT_READONLY_TOOLS}
    return PolicyBundle(version=1, default_action="deny", tools=tools)


def _load_from_file(path: Path) -> PolicyBundle:
    try:
        import yaml  # PyYAML
    except Exception as exc:
        raise PolicyLoadError(path, "dependency_unavailable") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError("tool policy must be a mapping")
        return PolicyBundle(**data)
    except PolicyLoadError:
        raise
    except Exception as exc:
        raise PolicyLoadError(path, "parse_error") from exc


@lru_cache
def load_tool_policy(
    policy_file: str = "",
    mode: str = "enforce",
    fail_closed: bool = False,
) -> PolicyBundle:
    """加载并缓存工具安全策略。

    缓存键均为可哈希标量，便于测试 cache_clear()。
    本地环境中空路径 / 文件不存在 / 解析失败会回退内置默认策略；
    fail_closed=True 时抛出 PolicyLoadError，由调用方明确识别并停止启动或请求。
    （mode 仅参与缓存键，实际裁决在 engine 依据 settings.tool_guard_mode 进行。）
    """
    path = Path(policy_file) if policy_file else Path("<not-configured>")
    if not policy_file or not path.is_file():
        error = PolicyLoadError(path, "missing")
        if fail_closed:
            raise error
        logger.warning("工具策略文件不存在，回退内置默认: %s", path)
        return _builtin_default_policy()
    try:
        return _load_from_file(path)
    except PolicyLoadError as exc:
        if fail_closed:
            raise
        logger.warning("工具策略加载失败，回退内置默认: %s", exc)
    return _builtin_default_policy()


def get_tool_policy(settings: Any) -> PolicyBundle:
    """按 settings 解析策略（从缓存取）。"""
    return load_tool_policy(
        getattr(settings, "tool_policy_file", "") or "",
        str(getattr(settings, "tool_guard_mode", "enforce") or "enforce"),
        str(getattr(settings, "app_env", "local") or "local").lower() == "production",
    )
