from __future__ import annotations

import getpass

from backend.src.slothbearflow_backend.security.auth import hash_password


def main() -> None:
    first = getpass.getpass("Password (minimum 12 characters): ")
    second = getpass.getpass("Confirm password: ")
    if first != second:
        raise SystemExit("Passwords do not match.")
    print(hash_password(first))


if __name__ == "__main__":
    main()
