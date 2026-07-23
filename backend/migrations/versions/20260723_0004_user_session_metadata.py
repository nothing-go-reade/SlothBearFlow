"""Persist user-owned session metadata for history restoration.

Revision ID: 20260723_0004
Revises: 20260717_0003
Create Date: 2026-07-23
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260723_0004"
down_revision: str | Sequence[str] | None = "20260717_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in (
        "ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS display_session_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''",
    ):
        op.execute(statement)
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_sessions_user_display
        ON agent_sessions (tenant_id, user_id, display_session_id)
        WHERE display_session_id <> ''
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_updated
        ON agent_sessions (tenant_id, user_id, updated_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_agent_sessions_user_updated")
    op.execute("DROP INDEX IF EXISTS uq_agent_sessions_user_display")
    op.execute("ALTER TABLE agent_sessions DROP COLUMN IF EXISTS tenant_id")
    op.execute("ALTER TABLE agent_sessions DROP COLUMN IF EXISTS user_id")
    op.execute("ALTER TABLE agent_sessions DROP COLUMN IF EXISTS display_session_id")
