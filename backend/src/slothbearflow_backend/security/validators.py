from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Tuple

from backend.src.slothbearflow_backend.security.schema import ArgConstraint


def validate_arg(value: Any, c: ArgConstraint) -> Tuple[bool, str]:
    """校验单个参数值是否满足约束。返回 (ok, reason)。

    allowlist 优先（enum/regex 校验 known-good），把 LLM 参数当不可信输入。
    """
    if c.type:
        ok, reason = _check_type(value, c.type)
        if not ok:
            return False, reason

    if isinstance(value, str):
        if c.max_len is not None and len(value) > c.max_len:
            return False, "argument too long (max_len=%s)" % c.max_len
        if c.min_len is not None and len(value) < c.min_len:
            return False, "argument too short (min_len=%s)" % c.min_len

    if c.enum is not None:
        allowed = [str(x) for x in c.enum]
        if str(value) not in allowed:
            return False, "argument not in allowed values"

    if c.regex is not None:
        try:
            if re.fullmatch(c.regex, str(value)) is None:
                return False, "argument failed regex allowlist"
        except re.error:
            # 策略正则本身写错 → 视为不阻断（配置问题不应误伤用户请求）。
            pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if c.min is not None and value < c.min:
            return False, "argument below min (%s)" % c.min
        if c.max is not None and value > c.max:
            return False, "argument above max (%s)" % c.max

    if c.path_within is not None and isinstance(value, str):
        ok, reason = _check_path_within(value, c.path_within)
        if not ok:
            return False, reason

    return True, ""


def _check_type(value: Any, type_name: str) -> Tuple[bool, str]:
    t = str(type_name).strip().lower()
    if t == "string":
        return (isinstance(value, str), "expected string")
    if t in ("integer", "int"):
        return (
            isinstance(value, int) and not isinstance(value, bool),
            "expected integer",
        )
    if t in ("number", "float"):
        return (
            isinstance(value, (int, float)) and not isinstance(value, bool),
            "expected number",
        )
    if t in ("boolean", "bool"):
        return (isinstance(value, bool), "expected boolean")
    # 未知类型名 → 不校验（宽松处理未知配置）。
    return True, ""


def _check_path_within(value: str, base: str) -> Tuple[bool, str]:
    """路径必须落在允许目录内（解析真实路径后判断，防 ../ 穿越）。"""
    try:
        base_p = Path(base).resolve()
        candidate = Path(value).resolve()
        try:
            within = candidate.is_relative_to(base_p)  # py>=3.9
        except AttributeError:  # pragma: no cover - py<3.9
            within = base_p == candidate or base_p in candidate.parents
        if not within:
            return False, "path escapes allowed directory"
    except Exception:
        return False, "invalid path"
    return True, ""
