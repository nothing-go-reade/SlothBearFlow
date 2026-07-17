from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from backend.src.slothbearflow_backend.security.execution import (
    ToolArgumentError,
    ToolInvocationError,
    current_cancellation_token,
    current_idempotency_key,
)
from backend.src.slothbearflow_backend.security.network import (
    is_literal_loopback_url,
    validate_outbound_url,
    validate_proxy_url,
)


MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_SUPPORTED_PROTOCOL_VERSIONS = frozenset({MCP_PROTOCOL_VERSION, "2025-03-26"})
_MAX_TOOLS_LIST_PAGES = 100
_MAX_MCP_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_MCP_TOOLS = 256
_MAX_MCP_DESCRIPTION_CHARS = 4096
_MAX_MCP_SCHEMA_BYTES = 64 * 1024
_MAX_MCP_TOOL_CONTENT_CHARS = 256 * 1024
_RESERVED_MCP_HEADERS = frozenset(
    {
        "accept",
        "authorization",
        "connection",
        "cookie",
        "content-length",
        "content-type",
        "host",
        "mcp-protocol-version",
        "mcp-session-id",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MCPError(RuntimeError):
    pass


class MCPServerConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    url: str
    enabled: bool = True
    bearer_token_env: str = ""
    tenant_bearer_token_envs: Dict[str, str] = Field(default_factory=dict)
    allowed_tenants: List[str] = Field(default_factory=list)
    allowed_roles: List[str] = Field(default_factory=list)
    allowed_scopes: List[str] = Field(default_factory=list)
    headers_env_json: str = ""

    @field_validator("url")
    @classmethod
    def _non_empty_url(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("MCP server URL is required")
        return str(value).strip()

    @field_validator("bearer_token_env", "headers_env_json")
    @classmethod
    def _valid_environment_name(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if normalized and _ENV_NAME_RE.fullmatch(normalized) is None:
            raise ValueError("MCP credential environment names must be identifiers")
        return normalized

    @field_validator("tenant_bearer_token_envs")
    @classmethod
    def _valid_tenant_environment_names(cls, value: Dict[str, str]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for tenant_id, env_name in value.items():
            tenant = str(tenant_id or "").strip()
            name = str(env_name or "").strip()
            if not tenant or _ENV_NAME_RE.fullmatch(name) is None:
                raise ValueError("tenant MCP credentials require named environment variables")
            normalized[tenant] = name
        return normalized


@dataclass(frozen=True)
class MCPToolDescriptor:
    server_name: str
    remote_name: str
    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class MCPToolResponse:
    content: str
    provenance: Dict[str, Any]

    def __str__(self) -> str:
        return self.content


class StreamableHttpMCPClient:
    def __init__(
        self,
        config: MCPServerConfig,
        *,
        allowed_hosts: List[str],
        timeout_sec: float,
        http_client: Optional[httpx.Client] = None,
        egress_proxy_url: str = "",
        require_external_proxy: bool = False,
    ) -> None:
        self.config = config
        self._allowed_hosts = tuple(allowed_hosts)
        self._timeout_sec = max(0.1, float(timeout_sec))
        self._external_target = not is_literal_loopback_url(config.url)
        self._egress_proxy_url = (
            validate_proxy_url(egress_proxy_url) if str(egress_proxy_url).strip() else ""
        )
        if require_external_proxy and self._external_target and not self._egress_proxy_url:
            raise MCPError("external MCP requires a controlled egress proxy")
        if require_external_proxy and self._external_target and http_client is not None:
            raise MCPError("external MCP cannot use a caller-supplied HTTP client in production")
        if http_client is not None:
            self._validate_supplied_client(http_client)
        self.url = self._validate_url()
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=self._timeout_sec,
            follow_redirects=False,
            trust_env=False,
            proxy=self._egress_proxy_url if self._external_target else None,
        )
        self._session_id = ""
        self._initialized = False
        self._protocol_version = MCP_PROTOCOL_VERSION
        self._request_id = 0
        self._lock = threading.RLock()

    @property
    def protocol_version(self) -> str:
        return self._protocol_version

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @staticmethod
    def _validate_supplied_client(http_client: httpx.Client) -> None:
        sensitive_headers = {
            "authorization",
            "cookie",
            "host",
            "mcp-protocol-version",
            "mcp-session-id",
            "proxy-authorization",
        }
        configured_headers = {str(key).lower() for key in http_client.headers.keys()}
        if configured_headers.intersection(sensitive_headers) or len(http_client.cookies):
            raise MCPError("caller-supplied MCP HTTP client contains reserved credentials")
        if getattr(http_client, "_auth", None) is not None:
            raise MCPError("caller-supplied MCP HTTP client cannot configure authentication")

    def _validate_url(self) -> str:
        return validate_outbound_url(
            self.config.url,
            allowed_hosts=self._allowed_hosts,
            require_https=True,
            # The exception remains constrained to exact loopback names that
            # are also present in the caller-provided host allowlist.
            allow_localhost=True,
            # An explicit proxy resolves the target and enforces the network
            # boundary, so the application must not resolve it a second time.
            resolve_dns=not (self._external_target and self._egress_proxy_url),
        )

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self._protocol_version,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if self.config.bearer_token_env:
            token = os.getenv(self.config.bearer_token_env, "").strip()
            if not token:
                raise MCPError("configured MCP bearer token is missing")
            if "\r" in token or "\n" in token:
                raise MCPError("configured MCP bearer token is invalid")
            headers["Authorization"] = "Bearer " + token
        if self.config.headers_env_json:
            raw = os.getenv(self.config.headers_env_json, "").strip()
            if raw:
                try:
                    values = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise MCPError(
                        "MCP headers environment value must contain valid JSON"
                    ) from exc
                if not isinstance(values, dict):
                    raise MCPError("MCP headers environment value must be a JSON object")
                for key, value in values.items():
                    header_name = str(key).strip()
                    lowered = header_name.lower()
                    if (
                        lowered in _RESERVED_MCP_HEADERS
                        or lowered.startswith("proxy-")
                        or lowered.startswith("x-forwarded-")
                    ):
                        raise MCPError(
                            f"custom MCP header is reserved: {header_name or '<empty>'}"
                        )
                    header_value = str(value)
                    if (
                        _HEADER_NAME_RE.fullmatch(header_name) is None
                        or "\r" in header_value
                        or "\n" in header_value
                    ):
                        raise MCPError("custom MCP header is invalid")
                    headers[header_name] = header_value
        return headers

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Any:
        with self._lock:
            request_id = self._next_id()
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
            response = self._post(payload, deadline=deadline)
            self._raise_for_status(response)
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                if len(session_id) > 1024 or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in session_id
                ):
                    raise MCPError("MCP session id is invalid")
                self._session_id = session_id
            body = self._decode_response(response)
            if body.get("id") != request_id:
                raise MCPError("MCP response id mismatch")
            if body.get("error"):
                error = body["error"]
                code = error.get("code") if isinstance(error, dict) else None
                message = (
                    str(error.get("message") or "MCP request failed")
                    if isinstance(error, dict)
                    else "MCP request failed"
                )
                if code == -32602:
                    raise ToolArgumentError(message)
                if code == -32603:
                    raise MCPError(message)
                raise ToolInvocationError(message)
            return body.get("result")

    def _notify(self, method: str, *, deadline: Optional[float] = None) -> None:
        with self._lock:
            payload = {"jsonrpc": "2.0", "method": method}
            response = self._post(payload, deadline=deadline)
            if response.status_code not in {200, 202, 204}:
                self._raise_for_status(response)

    def _post(
        self, payload: Dict[str, Any], *, deadline: Optional[float]
    ) -> httpx.Response:
        cancellation = current_cancellation_token()
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        try:
            response = self._client.post(
                self.url,
                headers=self._headers(),
                json=payload,
                timeout=self._remaining_timeout(deadline),
                follow_redirects=False,
            )
        except httpx.TimeoutException as exc:
            raise MCPError("MCP operation deadline exceeded") from exc
        except httpx.RequestError as exc:
            raise MCPError("MCP transport request failed") from exc
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        return response

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code in {400, 422}:
            raise ToolArgumentError("remote MCP rejected the tool arguments")
        if response.status_code >= 400:
            raise MCPError(f"MCP transport returned HTTP {response.status_code}")

    def _decode_response(self, response: httpx.Response) -> Dict[str, Any]:
        content_length = response.headers.get("content-length")
        try:
            declared_size = int(content_length) if content_length is not None else 0
        except ValueError as exc:
            raise MCPError("MCP response Content-Length is invalid") from exc
        if declared_size > _MAX_MCP_RESPONSE_BYTES or len(response.content) > _MAX_MCP_RESPONSE_BYTES:
            raise MCPError("MCP response exceeded the byte limit")
        content_type = response.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            body = response.json()
            if not isinstance(body, dict):
                raise MCPError("MCP response must be a JSON object")
            return body
        fallback = None
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            body = json.loads(line[5:].strip())
            if isinstance(body, dict):
                fallback = body
                if "id" in body:
                    return body
        if fallback is not None:
            return fallback
        raise MCPError("MCP SSE response did not contain a JSON-RPC message")

    def initialize(self, *, deadline: Optional[float] = None) -> None:
        with self._lock:
            if self._initialized:
                return
            if deadline is None:
                deadline = time.monotonic() + self._timeout_sec
            result = self._request(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "SlothBearFlow", "version": "0.2.0"},
                },
                deadline=deadline,
            )
            if not isinstance(result, dict):
                raise MCPError("MCP initialize result must be a JSON object")
            negotiated = str(result.get("protocolVersion") or "")
            if negotiated not in MCP_SUPPORTED_PROTOCOL_VERSIONS:
                raise MCPError(f"unsupported MCP protocol version: {negotiated or 'missing'}")
            self._protocol_version = negotiated
            try:
                self._notify("notifications/initialized", deadline=deadline)
            except Exception:
                self._protocol_version = MCP_PROTOCOL_VERSION
                raise
            self._initialized = True

    def list_tools(
        self, *, deadline: Optional[float] = None
    ) -> List[MCPToolDescriptor]:
        if deadline is None:
            deadline = time.monotonic() + self._timeout_sec
        self.initialize(deadline=deadline)
        descriptors: List[MCPToolDescriptor] = []
        seen_tools = set()
        seen_cursors = set()
        cursor: Optional[str] = None
        for _ in range(_MAX_TOOLS_LIST_PAGES):
            params = {"cursor": cursor} if cursor is not None else {}
            result = self._request("tools/list", params, deadline=deadline) or {}
            if not isinstance(result, dict):
                raise MCPError("MCP tools/list result must be a JSON object")
            tools = result.get("tools") or []
            if not isinstance(tools, list):
                raise MCPError("MCP tools/list tools must be an array")
            for item in tools:
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                remote_name = str(item["name"])
                if remote_name in seen_tools:
                    raise MCPError(f"duplicate MCP tool in paginated response: {remote_name}")
                input_schema = item.get("inputSchema") or {"type": "object"}
                if not isinstance(input_schema, dict):
                    raise MCPError(f"MCP tool inputSchema must be an object: {remote_name}")
                seen_tools.add(remote_name)
                if len(seen_tools) > _MAX_MCP_TOOLS:
                    raise MCPError("MCP tools/list exceeded the tool-count limit")
                description = str(item.get("description") or remote_name)
                if len(description) > _MAX_MCP_DESCRIPTION_CHARS:
                    raise MCPError(f"MCP tool description is too large: {remote_name}")
                schema_size = len(
                    json.dumps(input_schema, ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                )
                if schema_size > _MAX_MCP_SCHEMA_BYTES:
                    raise MCPError(f"MCP tool schema is too large: {remote_name}")
                descriptors.append(
                    MCPToolDescriptor(
                        server_name=self.config.name,
                        remote_name=remote_name,
                        name=f"mcp__{self.config.name}__{remote_name}",
                        description=description,
                        input_schema=dict(input_schema),
                    )
                )
            next_cursor = result.get("nextCursor")
            if next_cursor in (None, ""):
                return descriptors
            if not isinstance(next_cursor, str):
                raise MCPError("MCP tools/list nextCursor must be a string")
            if next_cursor in seen_cursors:
                raise MCPError("MCP tools/list returned a repeated cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        raise MCPError("MCP tools/list exceeded the pagination limit")

    def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        deadline: Optional[float] = None,
    ) -> MCPToolResponse:
        if deadline is None:
            deadline = time.monotonic() + self._timeout_sec
        self.initialize(deadline=deadline)
        params: Dict[str, Any] = {"name": name, "arguments": dict(arguments)}
        idempotency_key = current_idempotency_key()
        if idempotency_key:
            params["_meta"] = {
                "slothbearflow/idempotencyKey": idempotency_key,
            }
        result = self._request(
            "tools/call", params, deadline=deadline
        ) or {}
        if not isinstance(result, dict):
            raise MCPError("MCP tools/call result must be a JSON object")
        if result.get("isError"):
            raise ToolInvocationError("remote MCP tool returned an error")
        content_parts = []
        content_size = 0
        for item in result.get("content") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                part = str(item.get("text") or "")
            else:
                part = json.dumps(item, ensure_ascii=False)
            content_size += len(part)
            if content_size > _MAX_MCP_TOOL_CONTENT_CHARS:
                raise MCPError("MCP tool content exceeded the character limit")
            content_parts.append(part)
        return MCPToolResponse(
            content="\n".join(part for part in content_parts if part),
            provenance={
                "type": "mcp",
                "server": self.config.name,
                "remote_tool": name,
                "protocol_version": self._protocol_version,
                "call_id": idempotency_key or str(uuid.uuid4()),
            },
        )

    def _remaining_timeout(self, deadline: Optional[float]) -> float:
        if deadline is None:
            return self._timeout_sec
        remaining = deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise MCPError("MCP operation deadline exceeded")
        return min(self._timeout_sec, remaining)
