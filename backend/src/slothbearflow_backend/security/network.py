from __future__ import annotations

import ipaddress
import socket
from typing import Iterable
from urllib.parse import urlparse


class UnsafeOutboundUrl(ValueError):
    pass


_LOCALHOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _normalize_hostname(value: object) -> str:
    hostname = str(value or "").strip().lower().rstrip(".")
    try:
        return hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeOutboundUrl("URL hostname is invalid") from exc


def validate_outbound_url(
    url: str,
    *,
    allowed_hosts: Iterable[str],
    require_https: bool = False,
    allow_localhost: bool = False,
    resolve_dns: bool = True,
) -> str:
    raw_url = str(url or "").strip()
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeOutboundUrl("only http/https URLs are allowed")
    if parsed.username or parsed.password:
        raise UnsafeOutboundUrl("credentials in URLs are not allowed")
    try:
        hostname = _normalize_hostname(parsed.hostname)
    except ValueError as exc:
        raise UnsafeOutboundUrl("URL hostname is invalid") from exc
    if not hostname:
        raise UnsafeOutboundUrl("URL hostname is required")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeOutboundUrl("URL contains an invalid port") from exc
    allowlist = {_normalize_hostname(host) for host in allowed_hosts if str(host).strip()}
    if hostname not in allowlist:
        raise UnsafeOutboundUrl("outbound host is not in the allowlist")

    local_exception = allow_localhost and hostname in _LOCALHOST_NAMES
    if require_https and parsed.scheme != "https":
        if not (parsed.scheme == "http" and local_exception):
            raise UnsafeOutboundUrl("HTTPS is required for external outbound URLs")

    if not resolve_dns:
        return raw_url

    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        addresses = {str(item[4][0]).split("%", 1)[0] for item in answers}
    except (OSError, TypeError, ValueError) as exc:
        raise UnsafeOutboundUrl("outbound host cannot be resolved") from exc
    if not addresses:
        raise UnsafeOutboundUrl("outbound host cannot be resolved")

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise UnsafeOutboundUrl("outbound host returned an invalid address") from exc
        if local_exception:
            if ip.is_loopback:
                continue
        elif (
            ip.is_global
            and not ip.is_multicast
            and not ip.is_reserved
            and not getattr(ip, "is_site_local", False)
        ):
            continue
        raise UnsafeOutboundUrl("outbound host resolves to a blocked address")
    return raw_url


def is_literal_loopback_url(url: str) -> bool:
    try:
        hostname = _normalize_hostname(urlparse(str(url or "")).hostname)
    except ValueError:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_proxy_url(url: str) -> str:
    raw_url = str(url or "").strip()
    parsed = urlparse(raw_url)
    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise UnsafeOutboundUrl("egress proxy hostname is invalid") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise UnsafeOutboundUrl("egress proxy must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise UnsafeOutboundUrl("egress proxy URL cannot contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise UnsafeOutboundUrl("egress proxy URL must identify a proxy origin")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise UnsafeOutboundUrl("egress proxy URL contains an invalid port") from exc
    return raw_url
