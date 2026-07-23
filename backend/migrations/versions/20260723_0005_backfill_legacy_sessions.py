"""Backfill stable public aliases for legacy secure sessions.

Revision ID: 20260723_0005
Revises: 20260723_0004
Create Date: 2026-07-23
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260723_0005"
down_revision: str | Sequence[str] | None = "20260723_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        WITH owners AS (
            SELECT DISTINCT ON (session_id)
                session_id,
                user_id,
                tenant_id
            FROM agent_chat_turns
            WHERE user_id <> '' AND tenant_id <> ''
            ORDER BY session_id, created_at DESC, id DESC
        )
        UPDATE agent_sessions AS sessions
        SET user_id = owners.user_id,
            tenant_id = owners.tenant_id,
            display_session_id = CASE
                WHEN sessions.session_id LIKE 'secure:%'
                    THEN 'legacy-' || SUBSTRING(sessions.session_id FROM 8 FOR 16)
                ELSE sessions.session_id
            END
        FROM owners
        WHERE sessions.session_id = owners.session_id
          AND sessions.display_session_id = ''
        """
    )


def downgrade() -> None:
    # Legacy aliases may already be referenced by clients and remain valid metadata.
    pass
