from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


def _secret(prefix: str = "") -> str:
    return prefix + secrets.token_urlsafe(32)


def main() -> None:
    default_output = Path(__file__).resolve().parents[1] / ".env.observability"
    parser = argparse.ArgumentParser(
        description="Generate private credentials for the local observability stack."
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--email", default="admin@slothbearflow.local")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {output}; pass --force to replace it.")

    public_key = _secret("pk-lf-")
    secret_key = _secret("sk-lf-")
    admin_password = _secret()
    values = {
        "LANGFUSE_POSTGRES_PASSWORD": _secret(),
        "LANGFUSE_SALT": _secret(),
        "LANGFUSE_ENCRYPTION_KEY": secrets.token_hex(32),
        "LANGFUSE_CLICKHOUSE_PASSWORD": _secret(),
        "LANGFUSE_REDIS_PASSWORD": _secret(),
        "LANGFUSE_MINIO_ROOT_USER": "langfuse",
        "LANGFUSE_MINIO_ROOT_PASSWORD": _secret(),
        "LANGFUSE_NEXTAUTH_SECRET": _secret(),
        "LANGFUSE_VERSION": "3.217.0",
        "GRAFANA_ADMIN_USER": "admin",
        "GRAFANA_ADMIN_PASSWORD": _secret(),
        "METRICS_BEARER_TOKEN": _secret(),
        "LANGFUSE_INIT_ORG_ID": "slothbearflow",
        "LANGFUSE_INIT_ORG_NAME": "SlothBearFlow",
        "LANGFUSE_INIT_PROJECT_ID": "slothbearflow-agent",
        "LANGFUSE_INIT_PROJECT_NAME": "SlothBearFlow Agent",
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": public_key,
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": secret_key,
        "LANGFUSE_INIT_USER_EMAIL": args.email,
        "LANGFUSE_INIT_USER_NAME": "SlothBearFlow Admin",
        "LANGFUSE_INIT_USER_PASSWORD": admin_password,
        "OBSERVABILITY_ENABLED": "true",
        "PROMETHEUS_ENABLED": "true",
        "LANGFUSE_ENABLED": "true",
        "LANGFUSE_HOST": "http://langfuse-web:3000",
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_SECRET_KEY": secret_key,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
    )
    os.chmod(output, 0o600)
    print(f"Wrote {output} with mode 0600; credentials are stored only in that file.")


if __name__ == "__main__":
    main()
