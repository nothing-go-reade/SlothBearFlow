from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ArgConstraint(BaseModel):
    """单个工具参数的约束（allowlist 式校验，把 LLM 给的参数当不可信输入）。"""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    type: Optional[str] = None            # string | integer | number | boolean
    max_len: Optional[int] = None         # 字符串最大长度（防资源耗尽 / 注入）
    min_len: Optional[int] = None
    enum: Optional[List[str]] = None      # 取值白名单（known-good）
    regex: Optional[str] = None           # 全匹配正则（known-good）
    min: Optional[float] = None           # 数值下界
    max: Optional[float] = None           # 数值上界
    path_within: Optional[str] = None     # 路径必须落在该目录内（防穿越，SSRF 预留）

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in {"string", "integer", "int", "number", "float", "boolean", "bool"}:
            raise ValueError("unsupported argument type")
        return normalized

    @model_validator(mode="after")
    def _validate_ranges(self) -> "ArgConstraint":
        if self.min_len is not None and self.min_len < 0:
            raise ValueError("min_len cannot be negative")
        if self.max_len is not None and self.max_len <= 0:
            raise ValueError("max_len must be greater than zero")
        if self.min_len is not None and self.max_len is not None and self.min_len > self.max_len:
            raise ValueError("min_len cannot exceed max_len")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot exceed max")
        if self.regex is not None:
            import re

            re.compile(self.regex)
        return self


class ToolPolicy(BaseModel):
    """单个工具的准入策略。"""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    allow: bool = True
    # `class` 是 Python 关键字，字段名用 cls，YAML 里写 class:（靠 alias 映射）。
    cls: str = Field(default="read", alias="class")   # read | write
    max_calls_per_turn: Optional[int] = None
    requires_approval: bool = False
    timeout_sec: Optional[float] = None
    retry_attempts: Optional[int] = None
    retry_safe: bool = False
    allow_unknown_args: bool = False
    circuit_failure_threshold: Optional[int] = None
    circuit_recovery_sec: Optional[float] = None
    args: Dict[str, ArgConstraint] = Field(default_factory=dict)

    @field_validator("cls")
    @classmethod
    def _validate_class(cls, value: str) -> str:
        normalized = str(value or "read").strip().lower()
        if normalized not in {"read", "write", "network", "system"}:
            raise ValueError("tool class must be read, write, network, or system")
        return normalized

    @model_validator(mode="after")
    def _validate_execution_limits(self) -> "ToolPolicy":
        if self.max_calls_per_turn is not None and self.max_calls_per_turn <= 0:
            raise ValueError("max_calls_per_turn must be greater than zero")
        if self.timeout_sec is not None and self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be greater than zero")
        if self.retry_attempts is not None and not 0 <= self.retry_attempts <= 5:
            raise ValueError("retry_attempts must be between 0 and 5")
        if self.circuit_failure_threshold is not None and self.circuit_failure_threshold <= 0:
            raise ValueError("circuit_failure_threshold must be greater than zero")
        if self.circuit_recovery_sec is not None and self.circuit_recovery_sec <= 0:
            raise ValueError("circuit_recovery_sec must be greater than zero")
        if self.cls in {"write", "network", "system"} and not self.requires_approval:
            raise ValueError("write/network/system tools must require approval")
        if self.cls in {"write", "network", "system"} and self.allow_unknown_args:
            raise ValueError("write/network/system tools cannot allow unknown arguments")
        if self.cls in {"write", "network", "system"} and self.retry_safe:
            raise ValueError("side-effecting tools cannot be marked retry_safe")
        if (
            self.cls in {"write", "network", "system"}
            and self.retry_attempts not in {None, 0}
        ):
            raise ValueError("side-effecting tools cannot enable automatic retries")
        return self


class PolicyBundle(BaseModel):
    """整份工具安全策略（对应一个 YAML 文件）。"""

    model_config = {"populate_by_name": True, "extra": "forbid"}

    version: int = 1
    # 未列入 tools 的工具如何处置：deny（安全默认）| allow
    default_action: str = "deny"
    max_tool_calls_per_turn: Optional[int] = None
    tools: Dict[str, ToolPolicy] = Field(default_factory=dict)

    @field_validator("default_action")
    @classmethod
    def _validate_default_action(cls, value: str) -> str:
        normalized = str(value or "deny").strip().lower()
        if normalized not in {"allow", "deny"}:
            raise ValueError("default_action must be allow or deny")
        return normalized

    @model_validator(mode="after")
    def _validate_bundle(self) -> "PolicyBundle":
        if self.version <= 0:
            raise ValueError("policy version must be greater than zero")
        if self.max_tool_calls_per_turn is not None and self.max_tool_calls_per_turn <= 0:
            raise ValueError("max_tool_calls_per_turn must be greater than zero")
        return self

    def policy_for(self, name: str) -> Optional[ToolPolicy]:
        return self.tools.get(name)

    def allowed_tool_names(self) -> List[str]:
        return [name for name, tp in self.tools.items() if tp.allow]
