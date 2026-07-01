from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.src.slothbearflow_backend.security.schema import PolicyBundle
from backend.src.slothbearflow_backend.security.turn_state import record_and_check
from backend.src.slothbearflow_backend.security.validators import validate_arg

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


def _thread_whitelist():
    """复用后台复盘的线程级白名单（若已设置则由它独立裁决并短路策略文件）。"""
    try:
        from backend.src.slothbearflow_backend.learning.review_guard import (
            deny_message,
            get_thread_tool_whitelist,
        )

        return get_thread_tool_whitelist(), deny_message
    except Exception:  # pragma: no cover
        return None, None


def evaluate_tool_call(
    name: str,
    args: Optional[Dict[str, Any]] = None,
    *,
    settings: Any,
    policy: PolicyBundle,
    quota: bool = True,
) -> Decision:
    """工具调用的统一决策函数（PolicyGuardedTool 与 ReAct runtime 共用）。

    决策顺序：
      0. 线程级白名单（复盘上下文）—— 独立于 guard mode，始终强制；
      1. guard 模式门（off → 放行；仅作用于「策略文件」这一层）；
      2. 策略查表，未知工具走 default_action；
      3. allow 标志；
      4. requires_approval → headless 无 HITL，自动拒绝；
      5. 逐参数 allowlist 校验；
      6. 每回合配额。
    全程 try/except 降级：任何内部异常都放行（安全层绝不掀翻主链路）。
    """
    args = args or {}
    try:
        # 0) 线程级白名单优先短路：复盘上下文由它裁决，且不受 guard mode 影响，
        #    也不查策略文件（否则 default_action=deny 会误杀 save_memory/save_skill）。
        whitelist, deny_message = _thread_whitelist()
        if whitelist is not None:
            if name in whitelist:
                return Decision(True, "")
            reason = deny_message(name) if deny_message else "tool `%s` denied" % name
            return Decision(False, reason)

        # 1) guard 模式门（仅作用于策略文件层）
        mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce").lower()
        if mode == "off":
            return Decision(True, "")

        # 2) 策略查表；未知工具走 default_action
        tool_policy = policy.tools.get(name)
        if tool_policy is None:
            if str(policy.default_action or "deny").lower() == "allow":
                return Decision(True, "")
            return _maybe_log_only(
                mode, "tool `%s` not in allowlist (default deny)" % name
            )

        # 3) allow 标志
        if not tool_policy.allow:
            return _maybe_log_only(mode, "tool `%s` is disabled by policy" % name)

        # 4) requires_approval → 无人审通道，自动拒绝
        if tool_policy.requires_approval:
            return _maybe_log_only(
                mode, "tool `%s` requires approval; auto-denied in headless mode" % name
            )

        # 5) 逐参数校验（LLM 参数视为不可信输入）
        for arg_name, constraint in (tool_policy.args or {}).items():
            if arg_name in args:
                ok, reason = validate_arg(args[arg_name], constraint)
                if not ok:
                    return _maybe_log_only(
                        mode, "tool `%s` arg `%s` rejected: %s" % (name, arg_name, reason)
                    )

        # 6) 每回合配额
        if quota:
            global_limit = policy.max_tool_calls_per_turn
            if global_limit is None:
                global_limit = getattr(settings, "max_tool_calls_per_turn", None)
            ok, reason = record_and_check(
                name,
                per_tool_limit=tool_policy.max_calls_per_turn,
                global_limit=global_limit,
            )
            if not ok:
                return _maybe_log_only(mode, reason)

        return Decision(True, "")
    except Exception:  # pragma: no cover
        logger.exception("evaluate_tool_call 内部异常，放行降级: %s", name)
        return Decision(True, "")


def _maybe_log_only(mode: str, reason: str) -> Decision:
    """log 模式：记录违规但放行（dry-run 观察）；enforce 模式：真正拒绝。"""
    if mode == "log":
        logger.warning("[tool-guard log] would deny: %s", reason)
        return Decision(True, "")
    logger.info("[tool-guard] deny: %s", reason)
    return Decision(False, reason)
