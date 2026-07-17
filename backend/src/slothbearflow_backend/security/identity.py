from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from backend.src.slothbearflow_backend.observability.context import set_identity


@dataclass(frozen=True)
class Principal:
    user_id: str
    username: str
    tenant_id: str
    roles: FrozenSet[str] = field(default_factory=frozenset)
    scopes: FrozenSet[str] = field(default_factory=frozenset)
    anonymous: bool = False


_principal: contextvars.ContextVar[Optional[Principal]] = contextvars.ContextVar(
    "slothbearflow_principal", default=None
)


def bind_principal(principal: Principal) -> contextvars.Token:
    set_identity(user_id=principal.user_id, tenant_id=principal.tenant_id)
    return _principal.set(principal)


def reset_principal(token: contextvars.Token) -> None:
    _principal.reset(token)


def current_principal() -> Principal:
    return _principal.get() or Principal(
        user_id="anonymous",
        username="anonymous",
        tenant_id="local",
        roles=frozenset(),
        scopes=frozenset(),
        anonymous=True,
    )
