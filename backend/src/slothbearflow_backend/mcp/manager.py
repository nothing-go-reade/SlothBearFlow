from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from backend.src.slothbearflow_backend.mcp.client import (
    MCPServerConfig,
    MCPToolDescriptor,
    StreamableHttpMCPClient,
)
from backend.src.slothbearflow_backend.security.execution import current_tool_deadline


logger = logging.getLogger(__name__)
_cache_lock = threading.Lock()
_cache_entries: Dict[str, "_CacheEntry"] = {}
_cache_inflight: Dict[str, threading.Event] = {}
_cache_generation = 0
_status_by_scope: Dict[tuple[Any, ...], Dict[str, Any]] = {}
_status: Dict[str, Any] = {"enabled": False, "servers": [], "tool_count": 0}


@dataclass(frozen=True)
class MCPClientScope:
    tenant_id: str = "local"
    user_id: str = "local-user"
    scope: str = "default"
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _scope_part(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "user_id", _scope_part(self.user_id, "user_id"))
        object.__setattr__(self, "scope", _scope_part(self.scope, "scope"))
        object.__setattr__(self, "roles", _scope_values(self.roles, "roles"))
        object.__setattr__(self, "scopes", _scope_values(self.scopes, "scopes"))


@dataclass(frozen=True)
class _CacheEntry:
    cached_at: float
    tools: tuple[Any, ...]
    status: Dict[str, Any]


class MCPProxyTool(BaseTool):
    client: Any = None
    remote_name: str = ""
    provenance: Dict[str, Any] = Field(default_factory=dict)
    argument_aliases: Dict[str, str] = Field(default_factory=dict)

    def _parse_input(self, tool_input: Any, tool_call_id: Optional[str]) -> Any:
        if not isinstance(tool_input, dict) or self.args_schema is None:
            return super()._parse_input(tool_input, tool_call_id)
        if not isinstance(self.args_schema, type) or not issubclass(
            self.args_schema, BaseModel
        ):
            return super()._parse_input(tool_input, tool_call_id)
        validated = self.args_schema.model_validate(tool_input)
        return validated.model_dump(exclude_unset=True)

    def _run(self, **kwargs: Any) -> Any:
        arguments = {
            self.argument_aliases.get(key, key): value for key, value in kwargs.items()
        }
        return self.client.call_tool(
            self.remote_name,
            arguments,
            deadline=current_tool_deadline(),
        )

    async def _arun(self, **kwargs: Any) -> Any:
        import asyncio

        return await asyncio.to_thread(self._run, **kwargs)


def _schema_type(schema: Dict[str, Any]) -> Any:
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return Literal.__getitem__(tuple(enum_values))
    value = schema.get("type")
    if value == "string":
        return str
    if value == "integer":
        return int
    if value == "number":
        return float
    if value == "boolean":
        return bool
    if value == "array":
        item_schema = schema.get("items")
        item_type = _schema_type(item_schema) if isinstance(item_schema, dict) else Any
        return List[item_type]
    if value == "object":
        return dict
    return Any


def _args_model(
    descriptor: MCPToolDescriptor,
) -> Tuple[Type[BaseModel], Dict[str, str]]:
    schema = descriptor.input_schema or {}
    required = set(schema.get("required") or [])
    fields = {}
    aliases: Dict[str, str] = {}
    for index, (raw_name, raw_property_schema) in enumerate(
        (schema.get("properties") or {}).items()
    ):
        external_name = str(raw_name)
        property_schema = (
            dict(raw_property_schema) if isinstance(raw_property_schema, dict) else {}
        )
        internal_name = external_name
        if (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", internal_name) is None
            or hasattr(BaseModel, internal_name)
            or internal_name in fields
        ):
            internal_name = f"field_{index}"
        aliases[internal_name] = external_name
        annotation = _schema_type(property_schema)
        default = ... if external_name in required else None
        field_kwargs: Dict[str, Any] = {
            "alias": external_name,
            "description": str(property_schema.get("description") or ""),
        }
        for source, target in (
            ("minimum", "ge"),
            ("maximum", "le"),
            ("exclusiveMinimum", "gt"),
            ("exclusiveMaximum", "lt"),
            ("minLength", "min_length"),
            ("maxLength", "max_length"),
            ("minItems", "min_length"),
            ("maxItems", "max_length"),
        ):
            if property_schema.get(source) is not None:
                field_kwargs[target] = property_schema[source]
        if isinstance(property_schema.get("pattern"), str):
            field_kwargs["pattern"] = property_schema["pattern"]
        fields[internal_name] = (annotation, Field(default, **field_kwargs))
    model_name = "MCPArgs_" + re.sub(r"[^A-Za-z0-9_]", "_", descriptor.name)
    extra_mode = "forbid" if schema.get("additionalProperties", True) is False else "allow"
    return (
        create_model(
            model_name,
            __config__=ConfigDict(extra=extra_mode, populate_by_name=True),
            **fields,
        ),
        aliases,
    )


def _tool_from_descriptor(
    client: StreamableHttpMCPClient, descriptor: MCPToolDescriptor
) -> MCPProxyTool:
    args_model, aliases = _args_model(descriptor)
    return MCPProxyTool(
        name=descriptor.name,
        description=descriptor.description,
        args_schema=args_model,
        client=client,
        remote_name=descriptor.remote_name,
        argument_aliases=aliases,
        provenance={
            "type": "mcp",
            "server": descriptor.server_name,
            "remote_tool": descriptor.remote_name,
        },
    )


def _scope_part(value: object, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"MCP client {label} is required")
    if len(normalized) > 128 or any(
        ord(char) < 32 or ord(char) == 127 for char in normalized
    ):
        raise ValueError(f"MCP client {label} is invalid")
    return normalized


def _scope_values(values: Optional[Iterable[object]], label: str) -> frozenset[str]:
    if values is None:
        return frozenset()
    source = [values] if isinstance(values, str) else values
    return frozenset(_scope_part(item, label) for item in source)


def _status_key(client_scope: MCPClientScope) -> tuple[Any, ...]:
    return (
        client_scope.tenant_id,
        client_scope.user_id,
        client_scope.scope,
        tuple(sorted(client_scope.roles)),
        tuple(sorted(client_scope.scopes)),
    )


def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:  # noqa: BLE001
        logger.warning("failed to close MCP client", exc_info=True)


def _close_tools(tools: Iterable[Any]) -> None:
    clients = {
        id(client): client
        for tool in tools
        for client in [getattr(tool, "client", None)]
        if client is not None
    }
    for client in clients.values():
        _close_client(client)


def _resolve_scope(
    tenant_id: str,
    user_id: str,
    scope: str,
    roles: Iterable[str],
    scopes: Iterable[str],
    client_scope: Optional[MCPClientScope],
) -> MCPClientScope:
    return client_scope or MCPClientScope(
        tenant_id=tenant_id,
        user_id=user_id,
        scope=scope,
        roles=_scope_values(roles, "roles"),
        scopes=_scope_values(scopes, "scopes"),
    )


def _cache_key(
    settings: Any,
    server_rows: List[Any],
    allowlist: set[str],
    client_scope: MCPClientScope,
) -> str:
    payload = {
        "tenant_id": client_scope.tenant_id,
        "user_id": client_scope.user_id,
        "scope": client_scope.scope,
        "roles": sorted(client_scope.roles),
        "scopes": sorted(client_scope.scopes),
        "servers": server_rows,
        "tool_allowlist": sorted(allowlist),
        "allowed_hosts": sorted(
            str(item) for item in (getattr(settings, "mcp_allowed_hosts_json", None) or [])
        ),
        "timeout_sec": float(getattr(settings, "mcp_timeout_sec", 10.0)),
        "app_env": str(getattr(settings, "app_env", "local")),
        "egress_proxy_url": str(getattr(settings, "mcp_egress_proxy_url", "")),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _discover_tools(
    settings: Any,
    server_rows: List[Any],
    allowlist: set[str],
    client_scope: MCPClientScope,
) -> tuple[List[Any], Dict[str, Any]]:
    tools: List[Any] = []
    server_status = []
    discovery_deadline = time.monotonic() + max(
        0.1, float(getattr(settings, "mcp_timeout_sec", 10.0))
    )
    for row in server_rows:
        client: Optional[StreamableHttpMCPClient] = None
        try:
            remaining = discovery_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MCP discovery deadline exceeded")
            config = MCPServerConfig(**row)
            if not config.enabled:
                continue
            if config.allowed_tenants and client_scope.tenant_id not in set(
                config.allowed_tenants
            ):
                server_status.append(
                    {
                        "name": config.name,
                        "ready": False,
                        "skipped": "tenant_not_allowed",
                    }
                )
                continue
            if config.allowed_scopes and client_scope.scope not in set(
                config.allowed_scopes
            ) and not client_scope.scopes.intersection(config.allowed_scopes):
                server_status.append(
                    {
                        "name": config.name,
                        "ready": False,
                        "skipped": "scope_not_allowed",
                    }
                )
                continue
            if config.allowed_roles and not client_scope.roles.intersection(
                config.allowed_roles
            ):
                server_status.append(
                    {
                        "name": config.name,
                        "ready": False,
                        "skipped": "role_not_allowed",
                    }
                )
                continue
            if config.tenant_bearer_token_envs:
                token_env = str(
                    config.tenant_bearer_token_envs.get(client_scope.tenant_id) or ""
                ).strip()
                if not token_env:
                    server_status.append(
                        {
                            "name": config.name,
                            "ready": False,
                            "skipped": "tenant_credentials_missing",
                        }
                    )
                    continue
                config = config.model_copy(update={"bearer_token_env": token_env})
            client = StreamableHttpMCPClient(
                config,
                allowed_hosts=list(settings.mcp_allowed_hosts_json),
                timeout_sec=min(float(settings.mcp_timeout_sec), remaining),
                egress_proxy_url=str(
                    getattr(settings, "mcp_egress_proxy_url", "") or ""
                ),
                require_external_proxy=(
                    str(getattr(settings, "app_env", "local")) == "production"
                ),
            )
            descriptors = client.list_tools(deadline=discovery_deadline)
            selected = [item for item in descriptors if item.name in allowlist]
            selected_tools = [_tool_from_descriptor(client, item) for item in selected]
            tools.extend(selected_tools)
            if selected_tools:
                client = None
            server_status.append(
                {
                    "name": config.name,
                    "ready": True,
                    "discovered": len(descriptors),
                    "allowed": len(selected),
                }
            )
        except Exception as exc:  # noqa: BLE001
            name = str(row.get("name") or "invalid") if isinstance(row, dict) else "invalid"
            logger.warning("MCP discovery failed for %s: %s", name, exc)
            server_status.append({"name": name, "ready": False, "error": str(exc)})
        finally:
            _close_client(client)
    return tools, {
        "enabled": True,
        "servers": server_status,
        "tool_count": len(tools),
    }


def build_mcp_tools(
    settings: Any,
    *,
    tenant_id: str = "local",
    user_id: str = "local-user",
    scope: str = "default",
    roles: Iterable[str] = (),
    scopes: Iterable[str] = (),
    client_scope: Optional[MCPClientScope] = None,
) -> List[Any]:
    global _status
    resolved_scope = _resolve_scope(
        tenant_id,
        user_id,
        scope,
        roles,
        scopes,
        client_scope,
    )
    status_key = _status_key(resolved_scope)
    if not bool(getattr(settings, "mcp_enabled", False)):
        disabled_status = {"enabled": False, "servers": [], "tool_count": 0}
        with _cache_lock:
            _status = disabled_status
            _status_by_scope[status_key] = disabled_status
        return []
    server_rows = list(getattr(settings, "mcp_servers_json", None) or [])
    allowlist = set(getattr(settings, "mcp_tool_allowlist_json", None) or [])
    duplicate_names = _duplicate_server_names(server_rows)
    if duplicate_names:
        invalid_status = {
            "enabled": True,
            "servers": [],
            "tool_count": 0,
            "error": "duplicate_server_names",
            "duplicates": duplicate_names,
        }
        with _cache_lock:
            _status = invalid_status
            _status_by_scope[status_key] = invalid_status
        logger.error("MCP 配置包含重复 server name，已 fail closed: %s", duplicate_names)
        return []
    cache_key = _cache_key(settings, server_rows, allowlist, resolved_scope)
    now = time.monotonic()
    ttl = max(1.0, float(getattr(settings, "mcp_discovery_ttl_sec", 60.0)))
    with _cache_lock:
        cache_generation = _cache_generation
        cached = _cache_entries.get(cache_key)
        if cached is not None and now - cached.cached_at < ttl:
            _status = cached.status
            _status_by_scope[status_key] = cached.status
            return list(cached.tools)
        inflight = _cache_inflight.get(cache_key)
        discovery_owner = inflight is None
        if discovery_owner:
            inflight = threading.Event()
            _cache_inflight[cache_key] = inflight

    if not discovery_owner:
        assert inflight is not None
        if not inflight.wait(timeout=max(0.1, float(settings.mcp_timeout_sec))):
            timeout_status = {
                "enabled": True,
                "servers": [],
                "tool_count": 0,
                "error": "discovery_wait_timeout",
            }
            with _cache_lock:
                _status = timeout_status
                _status_by_scope[status_key] = timeout_status
            return []
        with _cache_lock:
            cached = _cache_entries.get(cache_key)
            if cached is not None:
                _status = cached.status
                _status_by_scope[status_key] = cached.status
                return list(cached.tools)
        return []

    # Client construction performs DNS resolution and list_tools performs HTTP I/O.
    # Both must remain outside the global cache lock.
    try:
        tools, status = _discover_tools(
            settings, server_rows, allowlist, resolved_scope
        )
        entry = _CacheEntry(
            cached_at=time.monotonic(),
            tools=tuple(tools),
            status=status,
        )
        with _cache_lock:
            if cache_generation != _cache_generation:
                stale = True
            else:
                stale = False
                _cache_entries[cache_key] = entry
                _status = entry.status
                _status_by_scope[status_key] = entry.status
        if stale:
            _close_tools(entry.tools)
            return []
        return list(entry.tools)
    finally:
        with _cache_lock:
            current = _cache_inflight.get(cache_key)
            if current is inflight:
                _cache_inflight.pop(cache_key, None)
                current.set()


def get_mcp_status(
    settings: Optional[Any] = None,
    *,
    tenant_id: str = "local",
    user_id: str = "local-user",
    scope: str = "default",
    roles: Iterable[str] = (),
    scopes: Iterable[str] = (),
    client_scope: Optional[MCPClientScope] = None,
) -> Dict[str, Any]:
    resolved_scope = _resolve_scope(
        tenant_id,
        user_id,
        scope,
        roles,
        scopes,
        client_scope,
    )
    if settings is not None:
        build_mcp_tools(settings, client_scope=resolved_scope)
    with _cache_lock:
        status = _status_by_scope.get(
            _status_key(resolved_scope),
            {"enabled": False, "servers": [], "tool_count": 0},
        )
        return copy.deepcopy(status)


def reset_mcp_cache() -> None:
    global _cache_generation, _status
    with _cache_lock:
        _cache_generation += 1
        entries = list(_cache_entries.values())
        _cache_entries.clear()
        for inflight in _cache_inflight.values():
            inflight.set()
        _cache_inflight.clear()
        _status_by_scope.clear()
        _status = {"enabled": False, "servers": [], "tool_count": 0}
    _close_tools(tool for entry in entries for tool in entry.tools)


def _duplicate_server_names(server_rows: Iterable[Any]) -> List[str]:
    seen = set()
    duplicates = set()
    for row in server_rows:
        if not isinstance(row, dict) or row.get("enabled", True) is False:
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    return sorted(duplicates)
