from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.src.slothbearflow_backend.security.schema import PolicyBundle
from backend.src.slothbearflow_backend.security.turn_state import (
    current_counts,
    record_and_check,
)
from backend.src.slothbearflow_backend.security.validators import validate_arg

logger = logging.getLogger(__name__)


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    approval_id: str = ""


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
      4. 逐参数 allowlist 校验；
      5. 每回合配额；
      6. requires_approval → 创建或消费一次性审批。
    enforce 模式下任何内部异常都拒绝；log/off 才允许受控降级。
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

        # 4) 逐参数校验（LLM 参数视为不可信输入）
        unknown_args = set(args).difference(tool_policy.args)
        if tool_policy.args and unknown_args and not tool_policy.allow_unknown_args:
            decision = _maybe_log_only(
                mode,
                "tool `%s` received unknown args: %s"
                % (name, ", ".join(sorted(str(item) for item in unknown_args))),
            )
            if not decision.allowed:
                return decision
        for arg_name, constraint in (tool_policy.args or {}).items():
            if arg_name in args:
                ok, reason = validate_arg(args[arg_name], constraint)
                if not ok:
                    decision = _maybe_log_only(
                        mode, "tool `%s` arg `%s` rejected: %s" % (name, arg_name, reason)
                    )
                    if not decision.allowed:
                        return decision

        quota_snapshot: Optional[Dict[str, int]] = None
        # 5) 每回合配额。审批先预留额度，pending 时再回滚。
        if quota:
            counts = current_counts()
            if tool_policy.requires_approval and counts is not None:
                quota_snapshot = dict(counts)
            global_limit = policy.max_tool_calls_per_turn
            if global_limit is None:
                global_limit = getattr(settings, "max_tool_calls_per_turn", None)
            ok, reason = record_and_check(
                name,
                per_tool_limit=tool_policy.max_calls_per_turn,
                global_limit=global_limit,
            )
            if not ok:
                decision = _maybe_log_only(mode, reason)
                if not decision.allowed:
                    return decision

        # 6) 审批 fingerprint 只能基于已校验、已通过配额的调用创建。
        if tool_policy.requires_approval:
            from backend.src.slothbearflow_backend.security.approval import approval_store

            try:
                approved, approval_id = approval_store.authorize_or_request(
                    tool_name=name,
                    args=args,
                    settings=settings,
                )
            except Exception:
                _restore_quota_snapshot(quota_snapshot)
                raise
            if not approved:
                _restore_quota_snapshot(quota_snapshot)
                return Decision(
                    False,
                    "tool `%s` requires approval (approval_id=%s)" % (
                        name,
                        approval_id,
                    ),
                    approval_id=approval_id,
                )

        return Decision(True, "")
    except Exception:  # pragma: no cover
        mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce").lower()
        logger.exception("evaluate_tool_call 内部异常，安全降级: %s", name)
        if mode in {"off", "log"}:
            return Decision(True, "")
        return Decision(False, "tool policy evaluation failed safely")


def _maybe_log_only(mode: str, reason: str) -> Decision:
    """log 模式：记录违规但放行（dry-run 观察）；enforce 模式：真正拒绝。"""
    if mode == "log":
        logger.warning("[tool-guard log] would deny: %s", reason)
        return Decision(True, "")
    logger.info("[tool-guard] deny: %s", reason)
    return Decision(False, reason)


def _restore_quota_snapshot(snapshot: Optional[Dict[str, int]]) -> None:
    if snapshot is None:
        return
    counts = current_counts()
    if counts is None:
        return
    counts.clear()
    counts.update(snapshot)
