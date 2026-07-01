from __future__ import annotations

from typing import Any, List, Optional

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.security import apply_tool_policy, get_tool_policy
from backend.src.slothbearflow_backend.tools.rag_tool import build_search_knowledge_tool
from backend.src.slothbearflow_backend.tools.session_tool import build_get_session_context_tool
from backend.src.slothbearflow_backend.tools.time_tool import get_current_time
from backend.src.slothbearflow_backend.tools.weather_tool import get_weather


def build_tools(
    vector_store: Optional[Any],
    *,
    chat_history: Optional[List[Any]] = None,
    settings: Optional[Settings] = None,
) -> List[Any]:
    settings = settings or get_settings()
    tools: List[Any] = [get_current_time, get_weather, build_get_session_context_tool(chat_history or [])]
    if settings.use_rag and not settings.skip_milvus and vector_store is not None:
        tools.append(build_search_knowledge_tool(vector_store))
    # 工具调用安全加固：按策略过滤 + 包裹（覆盖 AgentExecutor 与 ExplicitReActRuntime 两条路径）。
    # off 模式下 build_tools 不介入，工具原样返回（零开销、零行为变更）。
    mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce").lower()
    if mode != "off":
        tools = apply_tool_policy(tools, get_tool_policy(settings), settings)
    return tools
