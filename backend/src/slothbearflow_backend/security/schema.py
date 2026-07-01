from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ArgConstraint(BaseModel):
    """单个工具参数的约束（allowlist 式校验，把 LLM 给的参数当不可信输入）。"""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    type: Optional[str] = None            # string | integer | number | boolean
    max_len: Optional[int] = None         # 字符串最大长度（防资源耗尽 / 注入）
    min_len: Optional[int] = None
    enum: Optional[List[str]] = None      # 取值白名单（known-good）
    regex: Optional[str] = None           # 全匹配正则（known-good）
    min: Optional[float] = None           # 数值下界
    max: Optional[float] = None           # 数值上界
    path_within: Optional[str] = None     # 路径必须落在该目录内（防穿越，SSRF 预留）


class ToolPolicy(BaseModel):
    """单个工具的准入策略。"""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    allow: bool = True
    # `class` 是 Python 关键字，字段名用 cls，YAML 里写 class:（靠 alias 映射）。
    cls: str = Field(default="read", alias="class")   # read | write
    max_calls_per_turn: Optional[int] = None
    requires_approval: bool = False
    args: Dict[str, ArgConstraint] = Field(default_factory=dict)


class PolicyBundle(BaseModel):
    """整份工具安全策略（对应一个 YAML 文件）。"""

    model_config = {"populate_by_name": True, "extra": "ignore"}

    version: int = 1
    # 未列入 tools 的工具如何处置：deny（安全默认）| allow
    default_action: str = "deny"
    max_tool_calls_per_turn: Optional[int] = None
    tools: Dict[str, ToolPolicy] = Field(default_factory=dict)

    def policy_for(self, name: str) -> Optional[ToolPolicy]:
        return self.tools.get(name)

    def allowed_tool_names(self) -> List[str]:
        return [name for name, tp in self.tools.items() if tp.allow]
