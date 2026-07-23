from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from langchain_core.tools import tool
from pydantic import ValidationError


@pytest.fixture(autouse=True)
def _reset_security_caches() -> None:
    from backend.src.slothbearflow_backend.mcp import reset_mcp_cache
    from backend.src.slothbearflow_backend.security.loader import load_tool_policy

    reset_mcp_cache()
    load_tool_policy.cache_clear()
    yield
    reset_mcp_cache()
    load_tool_policy.cache_clear()


def _mcp_settings() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        mcp_enabled=True,
        mcp_servers_json=[{"name": "demo", "url": "https://mcp.example/mcp"}],
        mcp_tool_allowlist_json=["mcp__demo__lookup"],
        mcp_allowed_hosts_json=["mcp.example"],
        mcp_timeout_sec=1.0,
        mcp_discovery_ttl_sec=60.0,
        mcp_egress_proxy_url="",
        app_env="local",
    )


@pytest.mark.parametrize(
    ("policy_mode", "expected_visible"),
    [
        ("missing", False),
        ("deny", False),
        ("allow", True),
    ],
)
def test_registry_requires_explicit_mcp_policy_without_overriding_it(
    monkeypatch: pytest.MonkeyPatch,
    policy_mode: str,
    expected_visible: bool,
) -> None:
    import backend.src.slothbearflow_backend.tools.registry as registry
    from backend.src.slothbearflow_backend.config import get_settings
    from backend.src.slothbearflow_backend.security.schema import PolicyBundle, ToolPolicy

    @tool("mcp__demo__lookup")
    def remote_lookup(query: str) -> str:
        """Look up remote data."""
        return query

    policy_tools = {}
    if policy_mode != "missing":
        policy_tools["mcp__demo__lookup"] = ToolPolicy(
            allow=policy_mode == "allow",
            cls="network",
            requires_approval=True,
        )
    policy = PolicyBundle(default_action="deny", tools=policy_tools)
    observed_scope: dict[str, Any] = {}

    def fake_build_mcp_tools(settings: Any, **kwargs: Any) -> list[Any]:
        observed_scope.update(kwargs)
        return [remote_lookup]

    monkeypatch.setattr(registry, "build_mcp_tools", fake_build_mcp_tools)
    monkeypatch.setattr(registry, "get_tool_policy", lambda settings: policy)
    settings = get_settings().model_copy(
        update={
            "use_rag": False,
            "skip_milvus": True,
            "tool_guard_mode": "enforce",
        }
    )

    tools = registry.build_tools(
        None,
        settings=settings,
        rag_access_context=types.SimpleNamespace(
            tenant_id="tenant-a",
            user_id="alice",
            roles={"viewer"},
        ),
        mcp_scope="chat",
    )

    assert ("mcp__demo__lookup" in {item.name for item in tools}) is expected_visible
    if policy_mode != "missing":
        assert policy.tools["mcp__demo__lookup"].allow is expected_visible
    assert observed_scope == {
        "tenant_id": "tenant-a",
        "user_id": "alice",
        "scope": "chat",
        "roles": frozenset({"viewer"}),
        "scopes": frozenset(),
    }


def test_registry_propagates_bound_roles_and_scopes_to_mcp_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.tools.registry as registry
    from backend.src.slothbearflow_backend.config import get_settings
    from backend.src.slothbearflow_backend.security.identity import (
        Principal,
        bind_principal,
        reset_principal,
    )

    observed_scope: dict[str, Any] = {}

    def fake_build_mcp_tools(settings: Any, **kwargs: Any) -> list[Any]:
        observed_scope.update(kwargs)
        return []

    monkeypatch.setattr(registry, "build_mcp_tools", fake_build_mcp_tools)
    settings = get_settings().model_copy(
        update={
            "use_rag": False,
            "skip_milvus": True,
            "tool_guard_mode": "off",
        }
    )
    principal = Principal(
        user_id="alice",
        username="alice",
        tenant_id="tenant-a",
        roles=frozenset({"viewer"}),
        scopes=frozenset({"chat:write", "knowledge:read"}),
    )
    token = bind_principal(principal)
    try:
        async def build_in_worker() -> None:
            await asyncio.to_thread(
                registry.build_tools,
                None,
                settings=settings,
                rag_access_context=types.SimpleNamespace(
                    tenant_id="tenant-a",
                    user_id="alice",
                    roles={"viewer"},
                ),
                mcp_scope="chat",
            )

        asyncio.run(build_in_worker())
    finally:
        reset_principal(token)

    assert observed_scope["roles"] == frozenset({"viewer"})
    assert observed_scope["scopes"] == frozenset({"chat:write", "knowledge:read"})


def test_client_negotiates_protocol_and_follows_tools_list_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    requests: list[tuple[dict[str, Any], str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        requests.append((payload, request.headers.get("MCP-Protocol-Version", "")))
        method = payload.get("method")
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {}}
        elif method == "tools/list" and not payload.get("params", {}).get("cursor"):
            result = {
                "tools": [{"name": "first", "inputSchema": {"type": "object"}}],
                "nextCursor": "page-2",
            }
        elif method == "tools/list":
            assert payload["params"] == {"cursor": "page-2"}
            result = {
                "tools": [{"name": "second", "inputSchema": {"type": "object"}}]
            }
        else:
            result = {"content": [{"type": "text", "text": "ok"}]}
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            headers={"Mcp-Session-Id": "session-1"},
        )

    client = StreamableHttpMCPClient(
        MCPServerConfig(name="demo", url="http://localhost/mcp"),
        allowed_hosts=["localhost"],
        timeout_sec=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    descriptors = client.list_tools()
    monkeypatch.setattr(
        "backend.src.slothbearflow_backend.mcp.client.current_idempotency_key",
        lambda: "stable-call-key",
    )
    response = client.call_tool("first", {})

    assert [item.remote_name for item in descriptors] == ["first", "second"]
    assert response.provenance["protocol_version"] == "2025-03-26"
    assert [payload.get("method") for payload, _ in requests].count("tools/list") == 2
    post_initialize_versions = [
        version
        for payload, version in requests
        if payload.get("method") != "initialize"
    ]
    assert post_initialize_versions
    assert set(post_initialize_versions) == {"2025-03-26"}
    tool_call = next(payload for payload, _ in requests if payload.get("method") == "tools/call")
    assert tool_call["params"]["_meta"]["slothbearflow/idempotencyKey"] == "stable-call-key"


def test_client_rejects_unsupported_negotiated_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPError,
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"protocolVersion": "2099-01-01", "capabilities": {}},
            },
        )

    client = StreamableHttpMCPClient(
        MCPServerConfig(name="demo", url="http://localhost/mcp"),
        allowed_hosts=["localhost"],
        timeout_sec=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(MCPError, match="unsupported MCP protocol version"):
        client.initialize()


def test_client_list_tools_honors_expired_caller_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPError,
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    client = StreamableHttpMCPClient(
        MCPServerConfig(name="demo", url="http://localhost/mcp"),
        allowed_hosts=["localhost"],
        timeout_sec=1,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda request: None)),
    )

    with pytest.raises(MCPError, match="deadline exceeded"):
        client.list_tools(deadline=time.monotonic() - 0.01)


def test_custom_headers_preserve_protocol_and_scoped_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))
        ],
    )
    monkeypatch.setenv(
        "MCP_TEST_HEADERS",
        json.dumps(
            {
                "X-Custom": "kept",
            }
        ),
    )
    observed_headers: list[tuple[str, str, str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content or b"{}")
        observed_headers.append(
            (
                str(payload.get("method") or ""),
                request.headers.get("Mcp-Session-Id", ""),
                request.headers.get("MCP-Protocol-Version", ""),
                request.headers.get("Authorization", ""),
                request.headers.get("X-Custom", ""),
            )
        )
        if payload.get("method") == "notifications/initialized":
            return httpx.Response(202)
        result = (
            {"protocolVersion": "2025-03-26", "capabilities": {}}
            if payload.get("method") == "initialize"
            else {"tools": []}
        )
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
            headers={"Mcp-Session-Id": "scoped-session"},
        )

    client = StreamableHttpMCPClient(
        MCPServerConfig(
            name="demo",
            url="http://localhost/mcp",
            headers_env_json="MCP_TEST_HEADERS",
        ),
        allowed_hosts=["localhost"],
        timeout_sec=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.list_tools()

    initialize = next(item for item in observed_headers if item[0] == "initialize")
    initialized = next(
        item for item in observed_headers if item[0] == "notifications/initialized"
    )
    assert initialize[1:] == ("", "2025-06-18", "", "kept")
    assert initialized[1:] == ("scoped-session", "2025-03-26", "", "kept")


@pytest.mark.parametrize(
    "header_name",
    [
        "Authorization",
        "Cookie",
        "Host",
        "Mcp-Session-Id",
        "MCP-Protocol-Version",
        "Proxy-Authorization",
        "X-Forwarded-Host",
    ],
)
def test_custom_headers_reject_reserved_credential_and_routing_headers(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPError,
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    monkeypatch.setenv("MCP_TEST_HEADERS", json.dumps({header_name: "attacker"}))
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("reserved headers must fail before I/O")
        )
    ) as http_client:
        client = StreamableHttpMCPClient(
            MCPServerConfig(
                name="demo",
                url="http://127.0.0.1:1/mcp",
                headers_env_json="MCP_TEST_HEADERS",
            ),
            allowed_hosts=["127.0.0.1"],
            timeout_sec=1,
            http_client=http_client,
        )

        with pytest.raises(MCPError, match="reserved"):
            client.list_tools()


def test_supplied_http_client_cannot_inject_credentials() -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPError,
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    with httpx.Client(
        headers={"Authorization": "Bearer injected"},
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("credentialed client must fail before I/O")
        ),
    ) as http_client:
        with pytest.raises(MCPError, match="reserved credentials"):
            StreamableHttpMCPClient(
                MCPServerConfig(name="demo", url="http://127.0.0.1:1/mcp"),
                allowed_hosts=["127.0.0.1"],
                timeout_sec=1,
                http_client=http_client,
            )


def test_discovery_network_io_runs_without_cache_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager
    from backend.src.slothbearflow_backend.mcp.client import MCPToolDescriptor

    lock_was_available = threading.Event()

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def list_tools(self, *, deadline: float | None = None) -> list[MCPToolDescriptor]:
            assert deadline is not None
            acquired = manager._cache_lock.acquire(timeout=0.2)
            if acquired:
                manager._cache_lock.release()
                lock_was_available.set()
            return [
                MCPToolDescriptor(
                    server_name="demo",
                    remote_name="lookup",
                    name="mcp__demo__lookup",
                    description="lookup",
                    input_schema={"type": "object"},
                )
            ]

    monkeypatch.setattr(manager, "StreamableHttpMCPClient", FakeClient)
    manager.build_mcp_tools(_mcp_settings())
    assert lock_was_available.is_set()


def test_duplicate_mcp_server_names_fail_closed_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager

    class ForbiddenClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("duplicate server names must not start discovery")

    monkeypatch.setattr(manager, "StreamableHttpMCPClient", ForbiddenClient)
    settings = _mcp_settings()
    settings.mcp_servers_json = [
        {"name": "demo", "url": "https://mcp.example/one"},
        {"name": "demo", "url": "https://mcp.example/two"},
    ]

    assert manager.build_mcp_tools(settings) == []
    status = manager.get_mcp_status()
    assert status["error"] == "duplicate_server_names"
    assert status["duplicates"] == ["demo"]


def test_mcp_json_schema_constraints_and_aliases_are_preserved() -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager
    from backend.src.slothbearflow_backend.mcp.client import MCPToolDescriptor

    calls: list[dict[str, Any]] = []

    class Client:
        def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            *,
            deadline: float | None = None,
        ) -> str:
            calls.append({"name": name, "arguments": arguments, "deadline": deadline})
            return "ok"

    descriptor = MCPToolDescriptor(
        server_name="demo",
        remote_name="lookup",
        name="mcp__demo__lookup",
        description="lookup",
        input_schema={
            "type": "object",
            "properties": {
                "file-path": {"type": "string", "minLength": 3},
                "mode": {"type": "string", "enum": ["read", "metadata"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 5},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["file-path", "mode", "count"],
            "additionalProperties": False,
        },
    )
    tool = manager._tool_from_descriptor(Client(), descriptor)
    schema = tool.args_schema.model_json_schema(by_alias=True)

    assert set(schema["properties"]) == {"file-path", "mode", "count", "tags"}
    assert schema["properties"]["mode"]["enum"] == ["read", "metadata"]
    assert schema["properties"]["count"]["minimum"] == 1
    assert schema["properties"]["count"]["maximum"] == 5
    assert schema["properties"]["tags"]["items"]["type"] == "string"

    assert tool.invoke(
        {"file-path": "docs/a.md", "mode": "read", "count": 2, "tags": ["a"]}
    ) == "ok"
    assert calls[0]["arguments"] == {
        "file-path": "docs/a.md",
        "mode": "read",
        "count": 2,
        "tags": ["a"],
    }
    with pytest.raises(ValidationError):
        tool.invoke({"file-path": "x", "mode": "write", "count": 0, "extra": True})


def test_cache_reset_during_discovery_cannot_repopulate_or_evict_new_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager
    from backend.src.slothbearflow_backend.mcp.client import MCPToolDescriptor

    first_started = threading.Event()
    release_first = threading.Event()
    clients: list[Any] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.number = len(clients) + 1
            self.closed = False
            clients.append(self)

        def list_tools(self, *, deadline: float | None = None) -> list[MCPToolDescriptor]:
            if self.number == 1:
                first_started.set()
                assert release_first.wait(timeout=1)
            return [
                MCPToolDescriptor(
                    server_name="demo",
                    remote_name="lookup",
                    name="mcp__demo__lookup",
                    description="lookup",
                    input_schema={"type": "object"},
                )
            ]

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(manager, "StreamableHttpMCPClient", FakeClient)
    settings = _mcp_settings()

    with ThreadPoolExecutor(max_workers=2) as pool:
        stale_future = pool.submit(manager.build_mcp_tools, settings)
        assert first_started.wait(timeout=1)
        manager.reset_mcp_cache()
        fresh = manager.build_mcp_tools(settings)
        release_first.set()
        stale = stale_future.result(timeout=1)

    repeated = manager.build_mcp_tools(settings)
    assert stale == []
    assert fresh[0].client is repeated[0].client
    assert fresh[0].client is clients[1]
    assert clients[0].closed is True
    assert len(clients) == 2


def test_mcp_clients_are_cached_per_tenant_user_roles_and_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager
    from backend.src.slothbearflow_backend.mcp.client import MCPToolDescriptor

    clients: list[Any] = []

    class FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            clients.append(self)

        def list_tools(self, *, deadline: float | None = None) -> list[MCPToolDescriptor]:
            assert deadline is not None
            return [
                MCPToolDescriptor(
                    server_name="demo",
                    remote_name="lookup",
                    name="mcp__demo__lookup",
                    description="lookup",
                    input_schema={"type": "object"},
                )
            ]

        def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            return name

    monkeypatch.setattr(manager, "StreamableHttpMCPClient", FakeClient)
    settings = _mcp_settings()

    first = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-a",
        user_id="alice",
        scope="agent",
        roles={"viewer"},
        scopes={"chat:write"},
    )
    repeated = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-a",
        user_id="alice",
        scope="agent",
        roles={"viewer"},
        scopes={"chat:write"},
    )
    other_user = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-a",
        user_id="bob",
        scope="agent",
        roles={"viewer"},
        scopes={"chat:write"},
    )
    other_scope = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-a",
        user_id="alice",
        scope="background",
        roles={"viewer"},
        scopes={"chat:write"},
    )
    other_tenant = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-b",
        user_id="alice",
        scope="agent",
        roles={"viewer"},
        scopes={"chat:write"},
    )
    other_roles = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-a",
        user_id="alice",
        scope="agent",
        roles={"operator"},
        scopes={"chat:write"},
    )
    other_scopes = manager.build_mcp_tools(
        settings,
        tenant_id="tenant-a",
        user_id="alice",
        scope="agent",
        roles={"viewer"},
        scopes={"knowledge:write"},
    )

    assert first[0].client is repeated[0].client
    assert first[0].client is not other_scope[0].client
    assert first[0].client is not other_user[0].client
    assert first[0].client is not other_tenant[0].client
    assert first[0].client is not other_roles[0].client
    assert first[0].client is not other_scopes[0].client
    assert len(clients) == 6
    undiscovered_status = manager.get_mcp_status(
        tenant_id="tenant-a",
        user_id="alice",
        scope="agent",
        roles={"viewer"},
        scopes={"security:read"},
    )
    assert undiscovered_status == {"enabled": False, "servers": [], "tool_count": 0}


def test_mcp_resolves_tenant_scoped_credentials_and_rejects_other_tenants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager
    from backend.src.slothbearflow_backend.mcp.client import MCPToolDescriptor

    captured_token_envs: list[str] = []

    class FakeClient:
        def __init__(self, config: Any, **kwargs: Any) -> None:
            captured_token_envs.append(config.bearer_token_env)

        def list_tools(self, *, deadline: float | None = None) -> list[MCPToolDescriptor]:
            assert deadline is not None
            return [
                MCPToolDescriptor(
                    server_name="demo",
                    remote_name="lookup",
                    name="mcp__demo__lookup",
                    description="lookup",
                    input_schema={"type": "object"},
                )
            ]

    monkeypatch.setattr(manager, "StreamableHttpMCPClient", FakeClient)
    settings = _mcp_settings()
    settings.mcp_servers_json = [
        {
            "name": "demo",
            "url": "https://mcp.example/mcp",
            "allowed_tenants": ["tenant-a"],
            "allowed_scopes": ["viewer"],
            "tenant_bearer_token_envs": {"tenant-a": "TENANT_A_MCP_TOKEN"},
        }
    ]

    allowed = manager.build_mcp_tools(
        settings, tenant_id="tenant-a", scope="viewer"
    )
    blocked = manager.build_mcp_tools(
        settings, tenant_id="tenant-b", scope="viewer"
    )

    assert len(allowed) == 1
    assert blocked == []
    assert captured_token_envs == ["TENANT_A_MCP_TOKEN"]


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.1",
        "127.0.0.1",
        "100.64.0.1",
        "169.254.10.20",
        "224.0.0.1",
        "fc00::1",
        "fe80::1",
        "ff02::1",
    ],
)
def test_external_mcp_rejects_non_global_dns_answers(
    monkeypatch: pytest.MonkeyPatch,
    address: str,
) -> None:
    from backend.src.slothbearflow_backend.security.network import (
        UnsafeOutboundUrl,
        validate_outbound_url,
    )

    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (family, socket.SOCK_STREAM, 6, "", (address, 443))
        ],
    )
    with pytest.raises(UnsafeOutboundUrl, match="blocked address"):
        validate_outbound_url(
            "https://mcp.example/mcp",
            allowed_hosts=["mcp.example"],
            require_https=True,
            allow_localhost=True,
        )


def test_external_mcp_requires_https_and_localhost_exception_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.security.network import (
        UnsafeOutboundUrl,
        validate_outbound_url,
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 8080))
        ],
    )
    with pytest.raises(UnsafeOutboundUrl, match="HTTPS"):
        validate_outbound_url(
            "http://localhost:8080/mcp",
            allowed_hosts=["localhost"],
            require_https=True,
            allow_localhost=False,
        )
    assert (
        validate_outbound_url(
            "http://localhost:8080/mcp",
            allowed_hosts=["localhost"],
            require_https=True,
            allow_localhost=True,
        )
        == "http://localhost:8080/mcp"
    )
    with pytest.raises(UnsafeOutboundUrl, match="HTTPS"):
        validate_outbound_url(
            "http://mcp.example/mcp",
            allowed_hosts=["mcp.example"],
            require_https=True,
            allow_localhost=True,
        )


def test_mcp_client_wires_external_https_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPServerConfig,
        StreamableHttpMCPClient,
    )
    from backend.src.slothbearflow_backend.security.network import UnsafeOutboundUrl

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))
        ],
    )
    with pytest.raises(UnsafeOutboundUrl, match="HTTPS"):
        StreamableHttpMCPClient(
            MCPServerConfig(name="demo", url="http://mcp.example/mcp"),
            allowed_hosts=["mcp.example"],
            timeout_sec=1,
        )


def test_production_external_mcp_requires_proxy_without_local_dns_rebinding_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.src.slothbearflow_backend.mcp.client import (
        MCPError,
        MCPServerConfig,
        StreamableHttpMCPClient,
    )

    config = MCPServerConfig(name="demo", url="https://mcp.example/mcp")
    with pytest.raises(MCPError, match="egress proxy"):
        StreamableHttpMCPClient(
            config,
            allowed_hosts=["mcp.example"],
            timeout_sec=1,
            require_external_proxy=True,
        )

    dns_lookups = 0

    def unexpected_dns(*_args: Any, **_kwargs: Any) -> list[Any]:
        nonlocal dns_lookups
        dns_lookups += 1
        raise AssertionError("proxied MCP targets must not be resolved by the app")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    proxied_client = StreamableHttpMCPClient(
        config,
        allowed_hosts=["mcp.example"],
        timeout_sec=1,
        egress_proxy_url="http://127.0.0.1:3128",
        require_external_proxy=True,
    )
    proxied_client.close()

    assert dns_lookups == 0


def test_real_socket_mcp_e2e_sessions_pagination_deadline_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.mcp.manager as manager
    from backend.src.slothbearflow_backend.mcp.client import MCPError
    from backend.src.slothbearflow_backend.security.execution import execute_sync

    state: dict[str, Any] = {
        "lock": threading.Lock(),
        "next_session": 0,
        "sessions": set(),
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _send_json(
            self,
            status: int,
            payload: dict[str, Any] | None = None,
            *,
            session_id: str = "",
        ) -> None:
            body = json.dumps(payload).encode("utf-8") if payload is not None else b""
            try:
                self.send_response(status)
                if payload is not None:
                    self.send_header("Content-Type", "application/json")
                if session_id:
                    self.send_header("Mcp-Session-Id", session_id)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            method = str(payload.get("method") or "")
            session_id = self.headers.get("Mcp-Session-Id", "")
            request_record = {
                "method": method,
                "session_id": session_id,
                "authorization": self.headers.get("Authorization", ""),
                "cookie": self.headers.get("Cookie", ""),
                "host": self.headers.get("Host", ""),
                "custom": self.headers.get("X-MCP-Test", ""),
                "payload": payload,
            }
            with state["lock"]:
                state["requests"].append(request_record)

            if request_record["authorization"] != "Bearer socket-secret":
                self._send_json(401, {"error": "unauthorized"})
                return
            if method == "initialize":
                with state["lock"]:
                    state["next_session"] += 1
                    session_id = f"session-{state['next_session']}"
                    state["sessions"].add(session_id)
                self._send_json(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": payload["id"],
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                        },
                    },
                    session_id=session_id,
                )
                return
            with state["lock"]:
                valid_session = session_id in state["sessions"]
            if not valid_session:
                self._send_json(400, {"error": "invalid session"})
                return
            if method == "notifications/initialized":
                self._send_json(202)
                return
            if method == "tools/list":
                cursor = payload.get("params", {}).get("cursor")
                result = (
                    {
                        "tools": [
                            {
                                "name": "lookup",
                                "description": "lookup",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"],
                                },
                            }
                        ],
                        "nextCursor": "page-2",
                    }
                    if not cursor
                    else {
                        "tools": [
                            {
                                "name": "slow",
                                "description": "slow",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"delay": {"type": "number"}},
                                },
                            }
                        ]
                    }
                )
            elif method == "tools/call":
                params = payload.get("params", {})
                if params.get("name") == "slow":
                    time.sleep(float(params.get("arguments", {}).get("delay") or 0.15))
                text = f"{session_id}:{params.get('arguments', {}).get('query', 'slow')}"
                result = {"content": [{"type": "text", "text": text}]}
            else:
                self._send_json(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": payload.get("id"),
                        "error": {"code": -32601, "message": "unknown method"},
                    },
                )
                return
            self._send_json(
                200,
                {"jsonrpc": "2.0", "id": payload["id"], "result": result},
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    host, port = server.server_address
    monkeypatch.setenv("SOCKET_MCP_TOKEN", "socket-secret")
    monkeypatch.setenv("SOCKET_MCP_HEADERS", '{"X-MCP-Test":"preserved"}')
    settings = types.SimpleNamespace(
        app_env="local",
        mcp_enabled=True,
        mcp_servers_json=[
            {
                "name": "socket",
                "url": f"http://{host}:{port}/mcp",
                "tenant_bearer_token_envs": {"tenant-a": "SOCKET_MCP_TOKEN"},
                "headers_env_json": "SOCKET_MCP_HEADERS",
                "allowed_tenants": ["tenant-a"],
                "allowed_roles": ["viewer"],
                "allowed_scopes": ["chat:write"],
            }
        ],
        mcp_tool_allowlist_json=["mcp__socket__lookup", "mcp__socket__slow"],
        mcp_allowed_hosts_json=[host],
        mcp_timeout_sec=0.5,
        mcp_discovery_ttl_sec=60.0,
        mcp_egress_proxy_url="",
    )

    try:
        alice_tools = manager.build_mcp_tools(
            settings,
            tenant_id="tenant-a",
            user_id="alice",
            scope="agent",
            roles={"viewer"},
            scopes={"chat:write"},
        )
        alice_by_name = {item.name: item for item in alice_tools}
        alice_result = execute_sync(
            "tenant-a/socket/lookup",
            lambda: alice_by_name["mcp__socket__lookup"].invoke({"query": "alpha"}),
            timeout_sec=0.5,
            retries=3,
            retry_safe=True,
            failure_threshold=3,
            recovery_sec=1,
            idempotency_key="socket-call-stable",
            side_effecting=True,
        )

        bob_tools = manager.build_mcp_tools(
            settings,
            tenant_id="tenant-a",
            user_id="bob",
            scope="agent",
            roles={"viewer"},
            scopes={"chat:write"},
        )
        bob_by_name = {item.name: item for item in bob_tools}
        bob_result = bob_by_name["mcp__socket__lookup"].invoke({"query": "beta"})

        with pytest.raises(MCPError, match="deadline exceeded"):
            alice_by_name["mcp__socket__slow"].client.call_tool(
                "slow",
                {"delay": 0.15},
                deadline=time.monotonic() + 0.03,
            )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1)

    assert str(alice_result).startswith("session-1:alpha")
    assert str(bob_result).startswith("session-2:beta")
    assert alice_by_name["mcp__socket__lookup"].client is not bob_by_name[
        "mcp__socket__lookup"
    ].client
    requests = list(state["requests"])
    assert [item["method"] for item in requests].count("initialize") == 2
    assert [item["method"] for item in requests].count("tools/list") == 4
    assert [item["method"] for item in requests].count("tools/call") == 3
    assert all(
        item["session_id"] == ""
        for item in requests
        if item["method"] == "initialize"
    )
    cursors = [
        item["payload"].get("params", {}).get("cursor")
        for item in requests
        if item["method"] == "tools/list"
    ]
    assert cursors.count(None) == 2
    assert cursors.count("page-2") == 2
    assert all(item["authorization"] == "Bearer socket-secret" for item in requests)
    assert all(item["cookie"] == "" for item in requests)
    assert all(item["host"] == f"{host}:{port}" for item in requests)
    assert all(item["custom"] == "preserved" for item in requests)
    initialized_sessions = {
        item["session_id"]
        for item in requests
        if item["method"] == "notifications/initialized"
    }
    assert initialized_sessions == {"session-1", "session-2"}
    stable_call = next(
        item
        for item in requests
        if item["method"] == "tools/call"
        and item["payload"].get("params", {}).get("name") == "lookup"
        and item["payload"].get("params", {}).get("arguments", {}).get("query")
        == "alpha"
    )
    assert stable_call["payload"]["params"]["_meta"] == {
        "slothbearflow/idempotencyKey": "socket-call-stable"
    }


def test_unknown_username_runs_dummy_password_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import backend.src.slothbearflow_backend.security.auth as auth

    verified_hashes: list[str] = []

    def fake_verify(password: str, encoded: str) -> bool:
        verified_hashes.append(encoded)
        return False

    monkeypatch.setattr(auth, "verify_password", fake_verify)
    settings = types.SimpleNamespace(auth_users_json={}, audit_enabled=False)

    with pytest.raises(HTTPException) as exc_info:
        auth.authenticate_credentials("missing-user", "irrelevant-password", settings)

    assert exc_info.value.status_code == 401
    assert len(verified_hashes) == 1
    assert verified_hashes[0].startswith("pbkdf2_sha256$390000$")


def test_production_policy_failure_precedes_mcp_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import backend.src.slothbearflow_backend.tools.registry as registry
    from backend.src.slothbearflow_backend.config import get_settings
    from backend.src.slothbearflow_backend.security.loader import PolicyLoadError

    discovered = False

    def fake_build_mcp_tools(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal discovered
        discovered = True
        return []

    monkeypatch.setattr(registry, "build_mcp_tools", fake_build_mcp_tools)
    settings = get_settings().model_copy(
        update={
            "app_env": "production",
            "tool_guard_mode": "enforce",
            "tool_policy_file": str(tmp_path / "missing.yaml"),
            "use_rag": False,
            "skip_milvus": True,
        }
    )

    with pytest.raises(PolicyLoadError):
        registry.build_tools(None, settings=settings)
    assert discovered is False


@pytest.mark.parametrize(
    ("contents", "reason"),
    [
        (None, "missing"),
        ("tools: [", "parse_error"),
    ],
)
def test_production_policy_failure_is_typed_and_fails_closed(
    tmp_path: Path,
    contents: str | None,
    reason: str,
) -> None:
    from backend.src.slothbearflow_backend.security.loader import (
        PolicyLoadError,
        get_tool_policy,
    )

    policy_path = tmp_path / "tool-policy.yaml"
    if contents is not None:
        policy_path.write_text(contents, encoding="utf-8")
    settings = types.SimpleNamespace(
        app_env="production",
        tool_policy_file=str(policy_path),
        tool_guard_mode="enforce",
    )

    with pytest.raises(PolicyLoadError) as exc_info:
        get_tool_policy(settings)

    assert exc_info.value.reason == reason
    assert exc_info.value.path == policy_path
