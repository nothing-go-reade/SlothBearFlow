"""Add durable ingestion, manifest ACL, and memory deletion state.

Revision ID: 20260716_0002
Revises: 20260716_0001
Create Date: 2026-07-16
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260716_0002"
down_revision: str | Sequence[str] | None = "20260716_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in (
        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb",
        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ",
        "ALTER TABLE agent_knowledge_documents ADD COLUMN IF NOT EXISTS allowed_roles JSONB NOT NULL DEFAULT '[]'::jsonb",
        "ALTER TABLE agent_knowledge_documents ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE",
    ):
        op.execute(statement)
    op.execute(
        """
        WITH ranked AS (
            SELECT document_id, document_version,
                   ROW_NUMBER() OVER (
                       PARTITION BY document_id
                       ORDER BY indexed_at DESC, document_version DESC
                   ) AS row_number
            FROM agent_knowledge_documents
        )
        UPDATE agent_knowledge_documents AS document
        SET active = (ranked.row_number = 1)
        FROM ranked
        WHERE document.document_id = ranked.document_id
          AND document.document_version = ranked.document_version
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_ingest_jobs_claim
        ON agent_ingest_jobs (status, lease_expires_at, created_at)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_session_tombstones (
            session_id TEXT PRIMARY KEY,
            deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_session_tombstones")
    op.execute("DROP INDEX IF EXISTS idx_agent_ingest_jobs_claim")
    op.execute("ALTER TABLE agent_knowledge_documents DROP COLUMN IF EXISTS active")
    op.execute("ALTER TABLE agent_knowledge_documents DROP COLUMN IF EXISTS allowed_roles")
    op.execute("ALTER TABLE agent_ingest_jobs DROP COLUMN IF EXISTS lease_expires_at")
    op.execute("ALTER TABLE agent_ingest_jobs DROP COLUMN IF EXISTS attempts")
    op.execute("ALTER TABLE agent_ingest_jobs DROP COLUMN IF EXISTS payload")
