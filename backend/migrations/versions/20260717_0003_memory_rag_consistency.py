"""Add memory generations and durable RAG completion checkpoints.

Revision ID: 20260717_0003
Revises: 20260716_0002
Create Date: 2026-07-17
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260717_0003"
down_revision: str | Sequence[str] | None = "20260716_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for statement in (
        "ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0",
        "ALTER TABLE agent_chat_stream_events ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0",
        "ALTER TABLE agent_session_tombstones ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 1",
        "ALTER TABLE agent_session_tombstones ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS milvus_cleanup_completed BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS manifest_completed BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE agent_knowledge_documents ADD COLUMN IF NOT EXISTS chunking_contract JSONB NOT NULL DEFAULT '{}'::jsonb",
    ):
        op.execute(statement)
    op.execute(
        """
        UPDATE agent_ingest_jobs
        SET milvus_cleanup_completed = TRUE,
            manifest_completed = TRUE
        WHERE status = 'completed'
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'ck_agent_ingest_completed_checkpoints'
                  AND conrelid = 'agent_ingest_jobs'::regclass
            ) THEN
                ALTER TABLE agent_ingest_jobs
                ADD CONSTRAINT ck_agent_ingest_completed_checkpoints
                CHECK (
                    status <> 'completed'
                    OR (milvus_cleanup_completed AND manifest_completed)
                );
            END IF;
        END $$
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_chat_turns_session_generation_created
        ON agent_chat_turns (session_id, generation, created_at DESC)
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX IF EXISTS idx_agent_chat_turns_session_generation_created"
    )
    op.execute(
        "ALTER TABLE agent_ingest_jobs "
        "DROP CONSTRAINT IF EXISTS ck_agent_ingest_completed_checkpoints"
    )
    op.execute(
        "ALTER TABLE agent_knowledge_documents DROP COLUMN IF EXISTS chunking_contract"
    )
    op.execute(
        "ALTER TABLE agent_ingest_jobs DROP COLUMN IF EXISTS manifest_completed"
    )
    op.execute(
        "ALTER TABLE agent_ingest_jobs "
        "DROP COLUMN IF EXISTS milvus_cleanup_completed"
    )
    op.execute(
        "ALTER TABLE agent_session_tombstones DROP COLUMN IF EXISTS active"
    )
    op.execute(
        "ALTER TABLE agent_session_tombstones DROP COLUMN IF EXISTS generation"
    )
    op.execute(
        "ALTER TABLE agent_chat_stream_events DROP COLUMN IF EXISTS generation"
    )
    op.execute("ALTER TABLE agent_chat_turns DROP COLUMN IF EXISTS generation")
    op.execute("ALTER TABLE agent_sessions DROP COLUMN IF EXISTS generation")
