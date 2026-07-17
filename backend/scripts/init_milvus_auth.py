from __future__ import annotations

import os
from typing import Any, Callable


def _split_token(value: str, *, label: str) -> tuple[str, str]:
    username, separator, password = str(value or "").partition(":")
    if not separator or not username.strip() or not password:
        raise ValueError(f"{label} must use the username:password format")
    return username.strip(), password


def _close_quietly(client: Any) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _authenticated_client(
    client_factory: Callable[..., Any],
    *,
    uri: str,
    token: str,
    timeout: float,
) -> Any:
    client = client_factory(uri=uri, token=token, timeout=timeout)
    try:
        client.list_collections(timeout=timeout)
        return client
    except Exception:
        _close_quietly(client)
        raise


def initialize_milvus_auth(
    *,
    uri: str,
    desired_token: str,
    bootstrap_token: str,
    timeout: float = 10.0,
    client_factory: Callable[..., Any] | None = None,
) -> None:
    desired_user, desired_password = _split_token(desired_token, label="MILVUS_TOKEN")
    bootstrap_user, bootstrap_password = _split_token(
        bootstrap_token, label="MILVUS_BOOTSTRAP_TOKEN"
    )
    if desired_user != bootstrap_user:
        raise ValueError("Milvus bootstrap currently supports rotating the same user only")

    if client_factory is None:
        from pymilvus import MilvusClient

        client_factory = MilvusClient

    try:
        desired_client = _authenticated_client(
            client_factory,
            uri=uri,
            token=desired_token,
            timeout=timeout,
        )
    except Exception:  # noqa: BLE001
        bootstrap_client = _authenticated_client(
            client_factory,
            uri=uri,
            token=bootstrap_token,
            timeout=timeout,
        )
        try:
            bootstrap_client.update_password(
                desired_user,
                bootstrap_password,
                desired_password,
                reset_connection=True,
                timeout=timeout,
            )
        finally:
            _close_quietly(bootstrap_client)
        desired_client = _authenticated_client(
            client_factory,
            uri=uri,
            token=desired_token,
            timeout=timeout,
        )
    _close_quietly(desired_client)


def main() -> None:
    initialize_milvus_auth(
        uri=os.environ.get("MILVUS_URI", "http://milvus:19530"),
        desired_token=os.environ.get("MILVUS_TOKEN", ""),
        bootstrap_token=os.environ.get("MILVUS_BOOTSTRAP_TOKEN", ""),
        timeout=float(os.environ.get("MILVUS_AUTH_INIT_TIMEOUT_SEC", "10")),
    )
    print("Milvus target credentials verified.")


if __name__ == "__main__":
    main()
