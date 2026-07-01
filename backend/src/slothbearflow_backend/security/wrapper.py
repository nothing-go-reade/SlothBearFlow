from __future__ import annotations

import logging
from typing import Any, List

from langchain_core.tools import BaseTool

from backend.src.slothbearflow_backend.security.engine import evaluate_tool_call
from backend.src.slothbearflow_backend.security.schema import PolicyBundle
from backend.src.slothbearflow_backend.security.scrub import scrub_observation

logger = logging.getLogger(__name__)

# BaseTool.run 会把这些运行期参数混进 _run 的 kwargs，重建 arg dict 时要剔除。
_RESERVED_KWARGS = {"run_manager", "callbacks", "config"}

_VALIDATION_ERROR_MSG = "Tool input validation failed; please correct the arguments."


class PolicyGuardedTool(BaseTool):
    """把一个内部工具包裹进安全策略。

    两条执行路径都必经 BaseTool.run → _run（AgentExecutor 走 tool.run，
    ExplicitReActRuntime 走 tool.invoke→run），因此在 _run/_arun 里拦截可覆盖两者。
    复用内部工具的 name/description/args_schema → function-calling schema 不变。
    """

    inner_tool: Any = None
    policy: Any = None
    settings: Any = None

    def __init__(self, *, inner_tool: BaseTool, policy: PolicyBundle, settings: Any) -> None:
        super().__init__(
            name=inner_tool.name,
            description=inner_tool.description,
            args_schema=inner_tool.args_schema,
            # 类型错参数会在 _run 前触发 ValidationError → 返回短观测串而非抛异常，
            # 契合 AgentExecutor(handle_parsing_errors=True) 与 ReAct 的容错。
            handle_validation_error=_VALIDATION_ERROR_MSG,
            inner_tool=inner_tool,
            policy=policy,
            settings=settings,
        )

    def _clean_args(self, kwargs: dict) -> dict:
        return {k: v for k, v in kwargs.items() if k not in _RESERVED_KWARGS}

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        arg_dict = self._clean_args(kwargs)
        decision = evaluate_tool_call(
            self.name, arg_dict, settings=self.settings, policy=self.policy, quota=True
        )
        if not decision.allowed:
            return decision.reason
        result = self.inner_tool.invoke(arg_dict)
        return scrub_observation(result, self.settings)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        arg_dict = self._clean_args(kwargs)
        decision = evaluate_tool_call(
            self.name, arg_dict, settings=self.settings, policy=self.policy, quota=True
        )
        if not decision.allowed:
            return decision.reason
        result = await self.inner_tool.ainvoke(arg_dict)
        return scrub_observation(result, self.settings)


def apply_tool_policy(
    tools: List[Any], policy: PolicyBundle, settings: Any
) -> List[Any]:
    """按策略过滤 + 包裹工具列表。

    - enforce：丢弃 allow:false / 未列入(default deny) 的工具（模型看不到），再包裹存活工具；
    - log：不过滤（观察模式），仅包裹（wrapper 内部按 log 模式记录不阻断）；
    - off：理论上 build_tools 不会调用本函数，此处双保险直接原样返回。
    """
    mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce").lower()
    if mode == "off":
        return list(tools)

    result: List[Any] = []
    for tool in tools:
        name = getattr(tool, "name", None)
        if name is None:
            result.append(tool)
            continue
        if mode == "enforce":
            tp = policy.tools.get(name)
            if tp is not None and not tp.allow:
                logger.info("[tool-guard] filtered disabled tool: %s", name)
                continue
            if tp is None and str(policy.default_action or "deny").lower() != "allow":
                logger.info("[tool-guard] filtered non-allowlisted tool: %s", name)
                continue
        result.append(
            PolicyGuardedTool(inner_tool=tool, policy=policy, settings=settings)
        )
    return result
