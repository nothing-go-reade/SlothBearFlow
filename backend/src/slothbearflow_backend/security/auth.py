from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import time
from typing import Any, AsyncIterator, Callable, Dict, Iterable, Mapping

from fastapi import HTTPException, Request, status

from backend.src.slothbearflow_backend.config import get_settings
from backend.src.slothbearflow_backend.security.audit import audit_event
from backend.src.slothbearflow_backend.security.identity import (
    Principal,
    bind_principal,
    reset_principal,
)


ROLE_SCOPES = {
    "admin": {
        "chat:write",
        "knowledge:read",
        "knowledge:write",
        "memory:read",
        "memory:delete",
        "observability:read",
        "security:read",
        "security:approve",
    },
    "operator": {
        "chat:write",
        "knowledge:read",
        "knowledge:write",
        "memory:read",
        "observability:read",
        "security:read",
    },
    "viewer": {
        "chat:write",
        "knowledge:read",
        "memory:read",
        "observability:read",
    },
}
AUTH_COOKIE_NAME = "slothbearflow_session"

_DUMMY_PASSWORD_HASH = (
    "pbkdf2_sha256$390000$U2xvdGhCZWFyRmxvd0F1dGg$"
    "5Bv_jNEga9vqjop5JUi1Bm01S4u6Vf4V4OMv-WXJLSU"
)
_DUMMY_PASSWORD_ITERATIONS = 390_000
_DUMMY_PASSWORD_SALT = b"SlothBearFlowAuth"
_DUMMY_PASSWORD_DIGEST = base64.urlsafe_b64decode(
    "5Bv_jNEga9vqjop5JUi1Bm01S4u6Vf4V4OMv-WXJLSU="
)


def hash_password(password: str, *, iterations: int = 390_000) -> str:
    if len(password) < 12:
        raise ValueError("password must contain at least 12 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return "pbkdf2_sha256$%s$%s$%s" % (
        iterations,
        _b64(salt),
        _b64(digest),
    )


def verify_password(password: str, encoded: str) -> bool:
    valid_encoding = True
    try:
        algorithm, raw_iterations, raw_salt, raw_digest = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            raise ValueError("unsupported password algorithm")
        iterations = int(raw_iterations)
        if not 100_000 <= iterations <= 2_000_000:
            raise ValueError("invalid password iteration count")
        salt = _unb64(raw_salt)
        expected = _unb64(raw_digest)
    except (TypeError, ValueError):
        valid_encoding = False
        iterations = _DUMMY_PASSWORD_ITERATIONS
        salt = _DUMMY_PASSWORD_SALT
        expected = _DUMMY_PASSWORD_DIGEST
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return valid_encoding and hmac.compare_digest(actual, expected)


def authenticate_credentials(username: str, password: str, settings: Any) -> Principal:
    raw_row = (settings.auth_users_json or {}).get(username)
    row = dict(raw_row) if isinstance(raw_row, Mapping) else {}
    configured_hash = str(row.get("password_hash") or "")
    password_valid = verify_password(password, configured_hash or _DUMMY_PASSWORD_HASH)
    valid = bool(row and configured_hash and not row.get("disabled") and password_valid)
    if not valid:
        audit_event(
            settings,
            "auth.login_failed",
            actor=username[:128],
            metadata={"reason": "invalid_credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    principal = _principal_from_row(username, row)
    audit_event(
        settings,
        "auth.login_succeeded",
        actor=principal.user_id,
        tenant_id=principal.tenant_id,
    )
    return principal


def issue_access_token(principal: Principal, settings: Any) -> str:
    if len(settings.auth_secret) < 32:
        raise HTTPException(
            status_code=503,
            detail="Authentication is not configured with a strong signing secret.",
        )
    now = int(time.time())
    payload = {
        "iss": settings.auth_issuer,
        "sub": principal.user_id,
        "username": principal.username,
        "tenant_id": principal.tenant_id,
        "roles": sorted(principal.roles),
        "iat": now,
        "exp": now + int(settings.auth_token_ttl_sec),
        "jti": secrets.token_hex(16),
    }
    header = {"alg": "HS256", "typ": "JWT"}
    encoded_header = _b64(json.dumps(header, separators=(",", ":")).encode())
    encoded_payload = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = encoded_header + "." + encoded_payload
    signature = hmac.new(
        settings.auth_secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return signing_input + "." + _b64(signature)


def decode_access_token(token: str, settings: Any) -> Principal:
    if len(settings.auth_secret) < 32:
        raise _unauthorized()
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".", 2)
        header = json.loads(_unb64(encoded_header))
        if header != {"alg": "HS256", "typ": "JWT"}:
            raise ValueError("unexpected JWT header")
        signing_input = encoded_header + "." + encoded_payload
        expected = hmac.new(
            settings.auth_secret.encode(), signing_input.encode(), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, _unb64(encoded_signature)):
            raise ValueError("invalid signature")
        payload = json.loads(_unb64(encoded_payload))
        now = int(time.time())
        if payload.get("iss") != settings.auth_issuer or int(payload.get("exp") or 0) <= now:
            raise ValueError("expired or invalid issuer")
        roles = frozenset(str(value) for value in payload.get("roles") or [])
        username = str(payload.get("username") or payload["sub"])
        raw_row = (settings.auth_users_json or {}).get(username)
        if not isinstance(raw_row, Mapping) or raw_row.get("disabled"):
            raise ValueError("token user is no longer active")
        current = _principal_from_row(username, dict(raw_row))
        if (
            current.user_id != str(payload["sub"])
            or current.tenant_id != str(payload.get("tenant_id") or "default")
            or current.roles != roles
        ):
            raise ValueError("token identity or roles are stale")
        return current
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise _unauthorized()


def require_scopes(*required: str) -> Callable[[Request], AsyncIterator[Principal]]:
    async def dependency(request: Request) -> AsyncIterator[Principal]:
        settings = get_settings()
        authorization = request.headers.get("authorization", "")
        if authorization:
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise _unauthorized()
            principal = decode_access_token(token, settings)
        elif request.cookies.get(AUTH_COOKIE_NAME):
            principal = decode_access_token(
                str(request.cookies[AUTH_COOKIE_NAME]), settings
            )
        elif settings.auth_required:
            raise _unauthorized()
        else:
            if (
                str(getattr(settings, "app_env", "local")) != "test"
                and not _request_is_loopback(request)
            ):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Anonymous local mode only accepts loopback clients.",
                )
            roles = frozenset(settings.auth_local_roles_json)
            principal = Principal(
                user_id=settings.auth_local_user_id,
                username=settings.auth_local_user_id,
                tenant_id=settings.auth_local_tenant_id,
                roles=roles,
                scopes=_scopes_for_roles(roles),
                anonymous=True,
            )
        missing = set(required).difference(principal.scopes)
        if missing:
            audit_event(
                settings,
                "auth.authorization_denied",
                actor=principal.user_id,
                tenant_id=principal.tenant_id,
                metadata={"missing_scopes": sorted(missing)},
            )
            raise HTTPException(status_code=403, detail="Insufficient permissions.")
        token = bind_principal(principal)
        try:
            yield principal
        finally:
            reset_principal(token)

    return dependency


def namespace_session_id(session_id: str, principal: Principal, settings: Any) -> str:
    is_local_identity = (
        principal.user_id == settings.auth_local_user_id
        and principal.tenant_id == settings.auth_local_tenant_id
    )
    if not settings.auth_required and is_local_identity:
        return session_id
    namespace = f"{principal.tenant_id}:{principal.user_id}:{session_id}"
    return "secure:" + hashlib.sha256(namespace.encode()).hexdigest()


def _principal_from_row(username: str, row: Dict[str, Any]) -> Principal:
    roles = frozenset(str(value) for value in row.get("roles") or ["viewer"])
    return Principal(
        user_id=str(row.get("user_id") or username),
        username=username,
        tenant_id=str(row.get("tenant_id") or "default"),
        roles=roles,
        scopes=_scopes_for_roles(roles),
    )


def _scopes_for_roles(roles: Iterable[str]) -> frozenset[str]:
    scopes = set()
    for role in roles:
        scopes.update(ROLE_SCOPES.get(str(role), set()))
    return frozenset(scopes)


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _request_is_loopback(request: Request) -> bool:
    client = request.client
    host = str(client.host if client is not None else "")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
