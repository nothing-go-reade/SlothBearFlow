"""Create the initial SlothBearFlow persistence schema.

Revision ID: 20260716_0001
Revises:
Create Date: 2026-07-16
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "20260716_0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_sessions (
            session_id TEXT PRIMARY KEY,
            summary TEXT NOT NULL DEFAULT '',
            last_user_message TEXT NOT NULL DEFAULT '',
            last_assistant_message TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_chat_turns (
            id BIGSERIAL PRIMARY KEY,
            turn_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            assistant_message TEXT NOT NULL,
            raw_output TEXT NOT NULL DEFAULT '',
            response_source TEXT NOT NULL DEFAULT '',
            tools_used JSONB NOT NULL DEFAULT '[]'::jsonb,
            citations JSONB NOT NULL DEFAULT '[]'::jsonb,
            response_mode TEXT NOT NULL DEFAULT 'invoke',
            stream_format TEXT NOT NULL DEFAULT '',
            trace_id TEXT NOT NULL DEFAULT '',
            stop_reason TEXT NOT NULL DEFAULT '',
            steps INTEGER NOT NULL DEFAULT 0,
            tool_trace JSONB NOT NULL DEFAULT '[]'::jsonb,
            latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
            model TEXT NOT NULL DEFAULT '',
            executor TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL DEFAULT '',
            tenant_id TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    for statement in (
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS turn_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS response_mode TEXT NOT NULL DEFAULT 'invoke'",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS stream_format TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS trace_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS stop_reason TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS steps INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS tool_trace JSONB NOT NULL DEFAULT '[]'::jsonb",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS model TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS executor TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''",
    ):
        op.execute(statement)
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_chat_turns_turn_id
        ON agent_chat_turns (turn_id)
        WHERE turn_id <> ''
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_chat_turns_session_created
        ON agent_chat_turns (session_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_chat_stream_events (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_stream_events_session_created
        ON agent_chat_stream_events (session_id, created_at DESC)
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_ingest_jobs (
            job_id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'upload',
            text_length INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            error_detail TEXT NOT NULL DEFAULT '',
            tenant_id TEXT NOT NULL DEFAULT '',
            owner_id TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        "ALTER TABLE agent_ingest_jobs "
        "ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        "ALTER TABLE agent_ingest_jobs "
        "ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT ''"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_knowledge_documents (
            document_id TEXT NOT NULL,
            document_version TEXT NOT NULL,
            job_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL,
            tenant_id TEXT NOT NULL DEFAULT '',
            owner_id TEXT NOT NULL DEFAULT '',
            visibility TEXT NOT NULL DEFAULT 'tenant',
            chunk_count INTEGER NOT NULL DEFAULT 0,
            chunker_version TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (document_id, document_version)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_knowledge_tenant_indexed
        ON agent_knowledge_documents (tenant_id, indexed_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS agent_knowledge_documents")
    op.execute("DROP TABLE IF EXISTS agent_ingest_jobs")
    op.execute("DROP TABLE IF EXISTS agent_chat_stream_events")
    op.execute("DROP TABLE IF EXISTS agent_chat_turns")
    op.execute("DROP TABLE IF EXISTS agent_sessions")
