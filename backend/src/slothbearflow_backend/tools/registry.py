from __future__ import annotations

from typing import Any, Iterable, List, Optional

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.mcp import build_mcp_tools
from backend.src.slothbearflow_backend.rag.security import RagAccessContext
from backend.src.slothbearflow_backend.security import apply_tool_policy, get_tool_policy
from backend.src.slothbearflow_backend.security.identity import current_principal
from backend.src.slothbearflow_backend.tools.rag_tool import build_search_knowledge_tool
from backend.src.slothbearflow_backend.tools.session_tool import build_get_session_context_tool
from backend.src.slothbearflow_backend.tools.time_tool import get_current_time
from backend.src.slothbearflow_backend.tools.weather_tool import get_weather


def build_tools(
    vector_store: Optional[Any],
    *,
    chat_history: Optional[List[Any]] = None,
    settings: Optional[Settings] = None,
    rag_access_context: Optional[RagAccessContext] = None,
    mcp_tenant_id: Optional[str] = None,
    mcp_scope: Optional[str] = None,
    mcp_roles: Optional[Iterable[str]] = None,
    mcp_scopes: Optional[Iterable[str]] = None,
) -> List[Any]:
    settings = settings or get_settings()
    mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce").lower()
    policy = get_tool_policy(settings) if mode != "off" else None
    tools: List[Any] = [get_current_time, get_weather, build_get_session_context_tool(chat_history or [])]
    if settings.use_rag and not settings.skip_milvus and vector_store is not None:
        tools.append(
            build_search_knowledge_tool(
                vector_store,
                settings=settings,
                access_context=rag_access_context,
            )
        )
    resolved_tenant_id = (
        mcp_tenant_id
        if mcp_tenant_id is not None
        else str(
            getattr(rag_access_context, "tenant_id", "")
            or getattr(settings, "auth_local_tenant_id", "local")
        )
    )
    resolved_user_id = str(
        getattr(rag_access_context, "user_id", "")
        or getattr(settings, "auth_local_user_id", "local-user")
    )
    if mcp_scope is not None:
        resolved_scope = mcp_scope
    else:
        roles = sorted(str(role) for role in getattr(rag_access_context, "roles", set()))
        resolved_scope = "agent:" + (",".join(roles) if roles else "default")
    principal = current_principal()
    principal_matches = (
        principal.tenant_id == resolved_tenant_id
        and principal.user_id == resolved_user_id
    )
    raw_roles = (
        mcp_roles
        if mcp_roles is not None
        else (
            principal.roles
            if principal_matches
            else getattr(rag_access_context, "roles", set())
        )
    )
    raw_scopes = (
        mcp_scopes
        if mcp_scopes is not None
        else (principal.scopes if principal_matches else ())
    )
    resolved_roles = _identity_values(raw_roles)
    resolved_scopes = _identity_values(raw_scopes)
    mcp_tools = build_mcp_tools(
        settings,
        tenant_id=resolved_tenant_id,
        user_id=resolved_user_id,
        scope=resolved_scope,
        roles=resolved_roles,
        scopes=resolved_scopes,
    )
    tools.extend(mcp_tools)
    # 工具调用安全加固：按策略过滤 + 包裹（覆盖 AgentExecutor 与 ExplicitReActRuntime 两条路径）。
    # off 模式下 build_tools 不介入，工具原样返回（零开销、零行为变更）。
    if policy is not None:
        tools = apply_tool_policy(tools, policy, settings)
    return tools


def _identity_values(values: Iterable[str]) -> frozenset[str]:
    source = [values] if isinstance(values, str) else values
    return frozenset(str(value).strip() for value in source if str(value).strip())
