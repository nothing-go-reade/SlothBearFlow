from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.src.slothbearflow_backend import Settings, get_settings
from backend.src.slothbearflow_backend.memory.privacy import redact_memory_text
from backend.src.slothbearflow_backend.rag.security import (
    RagAccessContext,
    document_is_authorized,
    normalize_knowledge_acl,
)

logger = logging.getLogger(__name__)
_document_locks: dict[str, threading.RLock] = {}
_document_locks_guard = threading.Lock()


def _expected_migration_heads() -> tuple[str, ...]:
    try:
        from alembic.script import ScriptDirectory

        migrations_dir = Path(__file__).resolve().parents[3] / "migrations"
        return tuple(ScriptDirectory(str(migrations_dir)).get_heads())
    except Exception:  # noqa: BLE001
        logger.exception("无法读取 Alembic head")
        return ()


def _redact_persistent_value(value: Any, *, enabled: bool) -> Any:
    if isinstance(value, str):
        return redact_memory_text(value, enabled=enabled)
    if isinstance(value, dict):
        return {
            str(key): _redact_persistent_value(item, enabled=enabled)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_persistent_value(item, enabled=enabled) for item in value]
    return value


class PostgresPersistence:
    def __init__(self) -> None:
        self._schema_ready = False
        self._schema_ready_dsn = ""
        self._driver_checked = False
        self._psycopg = None
        self._pool_driver_checked = False
        self._pool_driver = None
        self._pool = None
        self._pool_dsn = ""
        self._pool_lock = threading.Lock()

    def is_enabled(self, settings: Optional[Settings] = None) -> bool:
        settings = settings or get_settings()
        return bool(
            getattr(settings, "enable_postgres_persistence", False)
            and str(getattr(settings, "postgres_dsn", "")).strip()
        )

    def _load_driver(self) -> Any:
        if self._driver_checked:
            return self._psycopg
        self._driver_checked = True
        try:
            import psycopg  # type: ignore

            self._psycopg = psycopg
        except ImportError:
            logger.warning("PostgreSQL 持久化未启用：缺少 psycopg 依赖")
            self._psycopg = None
        return self._psycopg

    def _get_connection(self, settings: Settings) -> Any:
        psycopg = self._load_driver()
        if psycopg is None:
            return None
        pool_driver = self._load_pool_driver()
        if pool_driver is not None:
            with self._pool_lock:
                if self._pool is None or self._pool_dsn != settings.postgres_dsn:
                    if self._pool is not None:
                        self._pool.close()
                    self._pool = pool_driver.ConnectionPool(
                        conninfo=settings.postgres_dsn,
                        min_size=settings.postgres_pool_min_size,
                        max_size=settings.postgres_pool_max_size,
                        kwargs={
                            "autocommit": True,
                            "connect_timeout": settings.postgres_connect_timeout,
                        },
                        open=True,
                    )
                    self._pool_dsn = settings.postgres_dsn
            return self._pool.connection()
        return psycopg.connect(
            settings.postgres_dsn,
            connect_timeout=settings.postgres_connect_timeout,
            autocommit=True,
        )

    def _load_pool_driver(self) -> Any:
        if self._pool_driver_checked:
            return self._pool_driver
        self._pool_driver_checked = True
        try:
            import psycopg_pool  # type: ignore

            self._pool_driver = psycopg_pool
        except ImportError:
            self._pool_driver = None
        return self._pool_driver

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None
                self._pool_dsn = ""
        self._schema_ready = False
        self._schema_ready_dsn = ""

    def ensure_schema(self, settings: Optional[Settings] = None) -> bool:
        settings = settings or get_settings()
        if not self.is_enabled(settings):
            return False
        if settings.app_env == "production":
            status = self.get_status(settings)
            ready = bool(status.get("ready"))
            self._schema_ready = ready
            self._schema_ready_dsn = settings.postgres_dsn if ready else ""
            return ready
        if self._schema_ready and self._schema_ready_dsn == settings.postgres_dsn:
            return True
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_sessions (
                            session_id TEXT PRIMARY KEY,
                            generation BIGINT NOT NULL DEFAULT 0,
                            summary TEXT NOT NULL DEFAULT '',
                            last_user_message TEXT NOT NULL DEFAULT '',
                            last_assistant_message TEXT NOT NULL DEFAULT '',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_chat_turns (
                            id BIGSERIAL PRIMARY KEY,
                            turn_id TEXT NOT NULL DEFAULT '',
                            session_id TEXT NOT NULL,
                            generation BIGINT NOT NULL DEFAULT 0,
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
                    cur.execute(
                        """
                        ALTER TABLE agent_chat_turns
                        ADD COLUMN IF NOT EXISTS response_mode TEXT NOT NULL DEFAULT 'invoke'
                        """
                    )
                    for statement in (
                        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS turn_id TEXT NOT NULL DEFAULT ''",
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
                        "ALTER TABLE agent_chat_turns ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0",
                    ):
                        cur.execute(statement)
                    cur.execute(
                        "ALTER TABLE agent_sessions ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0"
                    )
                    cur.execute(
                        """
                        CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_chat_turns_turn_id
                        ON agent_chat_turns (turn_id)
                        WHERE turn_id <> ''
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE agent_chat_turns
                        ADD COLUMN IF NOT EXISTS stream_format TEXT NOT NULL DEFAULT ''
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_chat_turns_session_created
                        ON agent_chat_turns (session_id, created_at DESC)
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_chat_stream_events (
                            id BIGSERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            generation BIGINT NOT NULL DEFAULT 0,
                            seq INTEGER NOT NULL,
                            event_type TEXT NOT NULL,
                            content TEXT NOT NULL DEFAULT '',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        "ALTER TABLE agent_chat_stream_events ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 0"
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_stream_events_session_created
                        ON agent_chat_stream_events (session_id, created_at DESC)
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_ingest_jobs (
                            job_id TEXT PRIMARY KEY,
                            source TEXT NOT NULL DEFAULT 'upload',
                            text_length INTEGER NOT NULL DEFAULT 0,
                            status TEXT NOT NULL,
                            error_detail TEXT NOT NULL DEFAULT '',
                            tenant_id TEXT NOT NULL DEFAULT '',
                            owner_id TEXT NOT NULL DEFAULT '',
                            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                            attempts INTEGER NOT NULL DEFAULT 0,
                            milvus_cleanup_completed BOOLEAN NOT NULL DEFAULT FALSE,
                            manifest_completed BOOLEAN NOT NULL DEFAULT FALSE,
                            lease_expires_at TIMESTAMPTZ,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT ''"
                    )
                    cur.execute(
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT ''"
                    )
                    for statement in (
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb",
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0",
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ",
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS milvus_cleanup_completed BOOLEAN NOT NULL DEFAULT FALSE",
                        "ALTER TABLE agent_ingest_jobs ADD COLUMN IF NOT EXISTS manifest_completed BOOLEAN NOT NULL DEFAULT FALSE",
                    ):
                        cur.execute(statement)
                    cur.execute(
                        """
                        UPDATE agent_ingest_jobs
                        SET milvus_cleanup_completed = TRUE,
                            manifest_completed = TRUE
                        WHERE status = 'completed'
                        """
                    )
                    cur.execute(
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
                                    OR (
                                        milvus_cleanup_completed
                                        AND manifest_completed
                                    )
                                );
                            END IF;
                        END $$
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_knowledge_documents (
                            document_id TEXT NOT NULL,
                            document_version TEXT NOT NULL,
                            job_id TEXT NOT NULL DEFAULT '',
                            source TEXT NOT NULL,
                            tenant_id TEXT NOT NULL DEFAULT '',
                            owner_id TEXT NOT NULL DEFAULT '',
                            visibility TEXT NOT NULL DEFAULT 'tenant',
                            allowed_roles JSONB NOT NULL DEFAULT '[]'::jsonb,
                            active BOOLEAN NOT NULL DEFAULT TRUE,
                            chunk_count INTEGER NOT NULL DEFAULT 0,
                            chunker_version TEXT NOT NULL DEFAULT '',
                            chunking_contract JSONB NOT NULL DEFAULT '{}'::jsonb,
                            embedding_model TEXT NOT NULL DEFAULT '',
                            indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (document_id, document_version)
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_knowledge_tenant_indexed
                        ON agent_knowledge_documents (tenant_id, indexed_at DESC)
                        """
                    )
                    for statement in (
                        "ALTER TABLE agent_knowledge_documents ADD COLUMN IF NOT EXISTS allowed_roles JSONB NOT NULL DEFAULT '[]'::jsonb",
                        "ALTER TABLE agent_knowledge_documents ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE",
                        "ALTER TABLE agent_knowledge_documents ADD COLUMN IF NOT EXISTS chunking_contract JSONB NOT NULL DEFAULT '{}'::jsonb",
                    ):
                        cur.execute(statement)
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_agent_ingest_jobs_claim
                        ON agent_ingest_jobs (status, lease_expires_at, created_at)
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_session_tombstones (
                            session_id TEXT PRIMARY KEY,
                            generation BIGINT NOT NULL DEFAULT 1,
                            active BOOLEAN NOT NULL DEFAULT TRUE,
                            deleted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        "ALTER TABLE agent_session_tombstones ADD COLUMN IF NOT EXISTS generation BIGINT NOT NULL DEFAULT 1"
                    )
                    cur.execute(
                        "ALTER TABLE agent_session_tombstones ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE"
                    )
            self._schema_ready = True
            self._schema_ready_dsn = settings.postgres_dsn
            logger.info("PostgreSQL 持久化表已就绪")
            return True
        except Exception:
            logger.exception("PostgreSQL 持久化初始化失败")
            return False

    def get_status(self, settings: Optional[Settings] = None) -> dict[str, Any]:
        settings = settings or get_settings()
        if not settings.enable_postgres_persistence:
            return {"enabled": False, "reason": "ENABLE_POSTGRES_PERSISTENCE=false"}
        if not str(settings.postgres_dsn).strip():
            return {"enabled": False, "reason": "POSTGRES_DSN is empty"}
        if self._load_driver() is None:
            return {"enabled": False, "reason": "psycopg is not installed"}
        try:
            with self._get_connection(settings) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                    required_tables = (
                        "agent_sessions",
                        "agent_chat_turns",
                        "agent_chat_stream_events",
                        "agent_ingest_jobs",
                        "agent_knowledge_documents",
                        "agent_session_tombstones",
                    )
                    missing = []
                    for table in required_tables:
                        cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                        if not cur.fetchone()[0]:
                            missing.append(table)
                    if missing:
                        return {
                            "enabled": True,
                            "ready": False,
                            "reason": "missing required tables: " + ", ".join(missing),
                        }
                    migration_revision = ""
                    cur.execute("SELECT to_regclass('public.alembic_version')")
                    has_alembic_table = bool(cur.fetchone()[0])
                    if has_alembic_table:
                        cur.execute("SELECT version_num FROM alembic_version LIMIT 1")
                        row = cur.fetchone()
                        migration_revision = str(row[0]) if row else ""
                    if settings.app_env == "production" and not migration_revision:
                        return {
                            "enabled": True,
                            "ready": False,
                            "reason": "Alembic migration revision is missing",
                        }
                    expected_heads: tuple[str, ...] = ()
                    if settings.app_env == "production":
                        expected_heads = _expected_migration_heads()
                        if not expected_heads:
                            return {
                                "enabled": True,
                                "ready": False,
                                "reason": "Alembic migration head cannot be determined",
                            }
                        if migration_revision not in expected_heads:
                            return {
                                "enabled": True,
                                "ready": False,
                                "reason": (
                                    f"database revision {migration_revision} is not at "
                                    f"required head {','.join(expected_heads)}"
                                ),
                                "migration_revision": migration_revision,
                                "expected_migration_heads": list(expected_heads),
                            }
            return {
                "enabled": True,
                "ready": True,
                "migration_revision": migration_revision or "runtime-schema",
                "expected_migration_heads": list(expected_heads),
            }
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "ready": False, "reason": str(exc)}

    def get_session_state(
        self,
        session_id: str,
        *,
        settings: Optional[Settings] = None,
    ) -> Optional[dict[str, Any]]:
        settings = settings or get_settings()
        if not self.is_enabled(settings):
            return {"generation": 0, "tombstoned": False, "persistent": False}
        if not self.ensure_schema(settings):
            return None
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT generation, active
                        FROM agent_session_tombstones
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    row = cur.fetchone()
            if row is None:
                return {"generation": 0, "tombstoned": False, "persistent": True}
            return {
                "generation": max(0, int(row[0] or 0)),
                "tombstoned": bool(row[1]),
                "persistent": True,
            }
        except Exception:
            logger.exception("PostgreSQL 读取 session state 失败: %s", session_id)
            return None

    @staticmethod
    def _context_generation(session_id: str) -> Optional[int]:
        try:
            from backend.src.slothbearflow_backend.memory.redis_memory import (
                current_session_generation,
            )

            return current_session_generation(session_id)
        except Exception:  # noqa: BLE001
            return None

    def _resolve_generation(
        self,
        session_id: str,
        generation: Optional[int],
        settings: Settings,
    ) -> Optional[int]:
        expected = generation
        if expected is None:
            expected = self._context_generation(session_id)
        if expected is not None:
            return max(0, int(expected))
        state = self.get_session_state(session_id, settings=settings)
        if state is None or bool(state.get("tombstoned")):
            return None
        return max(0, int(state.get("generation") or 0))

    def persist_chat_turn(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_message: str,
        raw_output: str,
        response_source: str,
        tools_used: Iterable[str],
        citations: Iterable[dict[str, Any]],
        response_mode: str = "invoke",
        stream_format: str = "",
        turn_id: str = "",
        trace_id: str = "",
        stop_reason: str = "",
        steps: int = 0,
        tool_trace: Iterable[dict[str, Any]] = (),
        latency_ms: float = 0.0,
        model: str = "",
        executor: str = "",
        prompt_version: str = "",
        user_id: str = "",
        tenant_id: str = "",
        generation: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        expected_generation = self._resolve_generation(
            session_id,
            generation,
            settings,
        )
        if expected_generation is None:
            return False
        try:
            redact_enabled = bool(settings.memory_redact_pii)
            user_message = redact_memory_text(user_message, enabled=redact_enabled)
            assistant_message = redact_memory_text(
                assistant_message, enabled=redact_enabled
            )
            raw_output = redact_memory_text(raw_output, enabled=redact_enabled)
            response_source = redact_memory_text(
                response_source, enabled=redact_enabled
            )
            tools_json = json.dumps(list(tools_used), ensure_ascii=False)
            citations_json = json.dumps(
                _redact_persistent_value(list(citations), enabled=redact_enabled),
                ensure_ascii=False,
            )
            tool_trace_json = json.dumps(
                _redact_persistent_value(list(tool_trace), enabled=redact_enabled),
                ensure_ascii=False,
            )
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (session_id,),
                    )
                    cur.execute(
                        "SELECT generation, active FROM agent_session_tombstones WHERE session_id = %s",
                        (session_id,),
                    )
                    state_row = cur.fetchone()
                    state_generation = int(state_row[0] or 0) if state_row else 0
                    state_tombstoned = bool(state_row[1]) if state_row else False
                    if state_tombstoned or state_generation != expected_generation:
                        cur.execute("ROLLBACK")
                        logger.info(
                            "跳过已删除或已换代 session 的迟到持久化: session_id=%s generation=%s current=%s",
                            session_id,
                            expected_generation,
                            state_generation,
                        )
                        return False
                    cur.execute(
                        """
                        INSERT INTO agent_sessions (
                            session_id,
                            generation,
                            last_user_message,
                            last_assistant_message
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            generation = EXCLUDED.generation,
                            last_user_message = EXCLUDED.last_user_message,
                            last_assistant_message = EXCLUDED.last_assistant_message,
                            updated_at = NOW()
                        WHERE agent_sessions.generation = EXCLUDED.generation
                        """,
                        (
                            session_id,
                            expected_generation,
                            user_message,
                            assistant_message,
                        ),
                    )
                    cur.execute(
                        """
                        INSERT INTO agent_chat_turns (
                            turn_id,
                            session_id,
                            generation,
                            user_message,
                            assistant_message,
                            raw_output,
                            response_source,
                            tools_used,
                            citations,
                            response_mode,
                            stream_format,
                            trace_id,
                            stop_reason,
                            steps,
                            tool_trace,
                            latency_ms,
                            model,
                            executor,
                            prompt_version,
                            user_id,
                            tenant_id
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s,
                            %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (turn_id) WHERE turn_id <> '' DO NOTHING
                        """,
                        (
                            turn_id,
                            session_id,
                            expected_generation,
                            user_message,
                            assistant_message,
                            raw_output,
                            response_source,
                            tools_json,
                            citations_json,
                            response_mode,
                            stream_format,
                            trace_id,
                            stop_reason,
                            steps,
                            tool_trace_json,
                            latency_ms,
                            model,
                            executor,
                            prompt_version,
                            user_id,
                            tenant_id,
                        ),
                    )
                    cur.execute("COMMIT")
            return True
        except Exception:
            logger.exception("PostgreSQL 持久化 chat turn 失败: session_id=%s", session_id)
            return False

    def persist_stream_events(
        self,
        *,
        session_id: str,
        events: Iterable[dict[str, Any]],
        generation: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        rows = list(events)
        if not rows:
            return True
        expected_generation = self._resolve_generation(
            session_id,
            generation,
            settings,
        )
        if expected_generation is None:
            return False
        redact_enabled = bool(settings.memory_redact_pii)
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (session_id,),
                    )
                    cur.execute(
                        "SELECT generation, active FROM agent_session_tombstones WHERE session_id = %s",
                        (session_id,),
                    )
                    state_row = cur.fetchone()
                    state_generation = int(state_row[0] or 0) if state_row else 0
                    state_tombstoned = bool(state_row[1]) if state_row else False
                    if state_tombstoned or state_generation != expected_generation:
                        cur.execute("ROLLBACK")
                        return False
                    for idx, event in enumerate(rows):
                        cur.execute(
                            """
                            INSERT INTO agent_chat_stream_events (
                                session_id,
                                generation,
                                seq,
                                event_type,
                                content
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                session_id,
                                expected_generation,
                                int(event.get("seq", idx)),
                                str(event.get("event_type") or "chunk"),
                                redact_memory_text(
                                    str(event.get("content") or ""),
                                    enabled=redact_enabled,
                                ),
                            ),
                        )
                    cur.execute("COMMIT")
            return True
        except Exception:
            logger.exception(
                "PostgreSQL 持久化 stream events 失败: session_id=%s", session_id
            )
            return False

    def load_session_snapshot(
        self,
        *,
        session_id: str,
        turn_limit: int,
        settings: Optional[Settings] = None,
    ) -> dict[str, Any]:
        settings = settings or get_settings()

        if not self.ensure_schema(settings) or turn_limit <= 0:
            return {"messages": [], "summary": "", "generation": 0}

        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return {"messages": [], "summary": "", "generation": 0}
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute(
                        "SELECT generation, active FROM agent_session_tombstones WHERE session_id = %s",
                        (session_id,),
                    )
                    state_row = cur.fetchone()
                    generation = int(state_row[0] or 0) if state_row else 0
                    if state_row and bool(state_row[1]):
                        cur.execute("ROLLBACK")
                        return {
                            "messages": [],
                            "summary": "",
                            "generation": generation,
                        }
                    cur.execute(
                        """
                        SELECT summary
                        FROM agent_sessions
                        WHERE session_id = %s AND generation = %s
                        """,
                        (session_id, generation),
                    )
                    row = cur.fetchone()
                    summary = str(row[0]) if row and row[0] else ""
                    cur.execute(
                        """
                        SELECT turn_id, user_message, assistant_message
                        FROM (
                            SELECT id, turn_id, user_message, assistant_message
                            FROM agent_chat_turns
                            WHERE session_id = %s AND generation = %s
                            ORDER BY id DESC
                            LIMIT %s
                        ) t
                        ORDER BY id ASC
                        """,
                        (session_id, generation, turn_limit),
                    )
                    rows = cur.fetchall() or []
            redact_enabled = bool(settings.memory_redact_pii)
            summary = redact_memory_text(summary, enabled=redact_enabled)
            messages = []
            for turn_id, user_message, assistant_message in rows:
                messages.append(
                    {
                        "role": "user",
                        "content": redact_memory_text(
                            str(user_message or ""), enabled=redact_enabled
                        ),
                        "turn_id": str(turn_id or ""),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": redact_memory_text(
                            str(assistant_message or ""), enabled=redact_enabled
                        ),
                        "turn_id": str(turn_id or ""),
                    }
                )
            return {
                "messages": messages,
                "summary": summary,
                "generation": generation,
            }
        except Exception:
            logger.exception("PostgreSQL 读取 session 快照失败: session_id=%s", session_id)
            return {"messages": [], "summary": "", "generation": 0}

    def persist_summary(
        self,
        session_id: str,
        summary: str,
        settings: Optional[Settings] = None,
        *,
        generation: Optional[int] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        expected_generation = self._resolve_generation(
            session_id,
            generation,
            settings,
        )
        if expected_generation is None:
            return False
        summary = redact_memory_text(
            summary, enabled=bool(settings.memory_redact_pii)
        )
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (session_id,),
                    )
                    cur.execute(
                        "SELECT generation, active FROM agent_session_tombstones WHERE session_id = %s",
                        (session_id,),
                    )
                    state_row = cur.fetchone()
                    state_generation = int(state_row[0] or 0) if state_row else 0
                    state_tombstoned = bool(state_row[1]) if state_row else False
                    if state_tombstoned or state_generation != expected_generation:
                        cur.execute("ROLLBACK")
                        return False
                    cur.execute(
                        """
                        INSERT INTO agent_sessions (session_id, generation, summary)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            generation = EXCLUDED.generation,
                            summary = EXCLUDED.summary,
                            updated_at = NOW()
                        WHERE agent_sessions.generation = EXCLUDED.generation
                        """,
                        (session_id, expected_generation, summary),
                    )
                    cur.execute("COMMIT")
            return True
        except Exception:
            logger.exception("PostgreSQL 持久化摘要失败: session_id=%s", session_id)
            return False

    def persist_ingest_job(
        self,
        *,
        job_id: str,
        source: str,
        text_length: int,
        status: str,
        error_detail: str = "",
        tenant_id: str = "",
        owner_id: str = "",
        payload: Optional[dict[str, Any]] = None,
        milvus_cleanup_completed: Optional[bool] = None,
        manifest_completed: Optional[bool] = None,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        try:
            payload_json = json.dumps(payload or {}, ensure_ascii=False)
            replace_payload = payload is not None
            terminal = status in {"completed", "skipped", "cancelled"}
            requeued = status == "queued"
            if status == "completed" and not (
                milvus_cleanup_completed is True and manifest_completed is True
            ):
                logger.error(
                    "拒绝未确认 cleanup/manifest 的 ingest 完成状态: job_id=%s",
                    job_id,
                )
                return False
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_ingest_jobs (
                            job_id,
                            source,
                            text_length,
                            status,
                            error_detail,
                            tenant_id,
                            owner_id,
                            payload,
                            milvus_cleanup_completed,
                            manifest_completed,
                            lease_expires_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                            %s, %s, NULL
                        )
                        ON CONFLICT (job_id) DO UPDATE SET
                            source = EXCLUDED.source,
                            text_length = EXCLUDED.text_length,
                            status = EXCLUDED.status,
                            error_detail = EXCLUDED.error_detail,
                            tenant_id = EXCLUDED.tenant_id,
                            owner_id = EXCLUDED.owner_id,
                            payload = CASE
                                WHEN %s THEN EXCLUDED.payload
                                WHEN %s THEN '{}'::jsonb
                                ELSE agent_ingest_jobs.payload
                            END,
                            milvus_cleanup_completed = CASE
                                WHEN %s THEN EXCLUDED.milvus_cleanup_completed
                                ELSE agent_ingest_jobs.milvus_cleanup_completed
                            END,
                            manifest_completed = CASE
                                WHEN %s THEN EXCLUDED.manifest_completed
                                ELSE agent_ingest_jobs.manifest_completed
                            END,
                            lease_expires_at = CASE
                                WHEN %s OR %s THEN NULL
                                ELSE agent_ingest_jobs.lease_expires_at
                            END,
                            updated_at = NOW()
                        """,
                        (
                            job_id,
                            source,
                            text_length,
                            status,
                            error_detail,
                            tenant_id,
                            owner_id,
                            payload_json,
                            bool(milvus_cleanup_completed),
                            bool(manifest_completed),
                            replace_payload,
                            terminal,
                            milvus_cleanup_completed is not None,
                            manifest_completed is not None,
                            terminal,
                            requeued,
                        ),
                    )
            return True
        except Exception:
            logger.exception("PostgreSQL 持久化 ingest 任务失败: job_id=%s", job_id)
            return False

    def record_ingest_checkpoint(
        self,
        job_id: str,
        *,
        milvus_cleanup_completed: Optional[bool] = None,
        manifest_completed: Optional[bool] = None,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        if milvus_cleanup_completed is None and manifest_completed is None:
            return True
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE agent_ingest_jobs
                        SET milvus_cleanup_completed = CASE
                                WHEN %s THEN %s
                                ELSE milvus_cleanup_completed
                            END,
                            manifest_completed = CASE
                                WHEN %s THEN %s
                                ELSE manifest_completed
                            END,
                            updated_at = NOW()
                        WHERE job_id = %s
                          AND payload <> '{}'::jsonb
                        """,
                        (
                            milvus_cleanup_completed is not None,
                            bool(milvus_cleanup_completed),
                            manifest_completed is not None,
                            bool(manifest_completed),
                            job_id,
                        ),
                    )
                    return bool(cur.rowcount)
        except Exception:
            logger.exception("PostgreSQL 更新 ingest checkpoint 失败: job_id=%s", job_id)
            return False

    def complete_ingest_job(
        self,
        job_id: str,
        *,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE agent_ingest_jobs
                        SET status = 'completed',
                            error_detail = '',
                            payload = '{}'::jsonb,
                            lease_expires_at = NULL,
                            updated_at = NOW()
                        WHERE job_id = %s
                          AND payload <> '{}'::jsonb
                          AND milvus_cleanup_completed = TRUE
                          AND manifest_completed = TRUE
                        """,
                        (job_id,),
                    )
                    return bool(cur.rowcount)
        except Exception:
            logger.exception("PostgreSQL 完成 ingest outbox 失败: job_id=%s", job_id)
            return False

    def claim_ingest_job(
        self,
        job_id: str = "",
        *,
        settings: Optional[Settings] = None,
    ) -> Optional[dict[str, Any]]:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return None
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        WITH candidate AS (
                            SELECT job_id
                            FROM agent_ingest_jobs
                            WHERE payload <> '{}'::jsonb
                              AND (
                                  (
                                      status = 'queued'
                                      AND (
                                          lease_expires_at IS NULL
                                          OR lease_expires_at < NOW()
                                      )
                                  )
                                  OR (
                                      status = 'processing'
                                      AND lease_expires_at IS NOT NULL
                                      AND lease_expires_at < NOW()
                                  )
                              )
                              AND (%s = '' OR job_id = %s)
                            ORDER BY created_at ASC
                            FOR UPDATE SKIP LOCKED
                            LIMIT 1
                        )
                        UPDATE agent_ingest_jobs AS job
                        SET status = 'processing',
                            attempts = attempts + 1,
                            lease_expires_at = NOW() + INTERVAL '30 minutes',
                            updated_at = NOW()
                        FROM candidate
                        WHERE job.job_id = candidate.job_id
                        RETURNING job.job_id, job.source, job.payload,
                                  job.tenant_id, job.owner_id, job.attempts
                        """,
                        (job_id, job_id),
                    )
                    row = cur.fetchone()
            if row is None:
                return None
            payload = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
            result = dict(payload or {})
            result["type"] = "ingest"
            result["job_id"] = str(row[0])
            result.setdefault("source", str(row[1] or "upload"))
            metadata = dict(result.get("metadata") or {})
            metadata.setdefault("tenant_id", str(row[3] or ""))
            metadata.setdefault("owner_id", str(row[4] or ""))
            result["metadata"] = metadata
            result["attempts"] = int(row[5] or 0)
            return result
        except Exception:
            logger.exception("PostgreSQL claim ingest 任务失败: job_id=%s", job_id)
            return None

    def defer_ingest_job(
        self,
        job_id: str,
        *,
        error_detail: str,
        delay_sec: float,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE agent_ingest_jobs
                        SET status = 'queued',
                            error_detail = %s,
                            lease_expires_at = NOW() + (%s * INTERVAL '1 second'),
                            updated_at = NOW()
                        WHERE job_id = %s
                          AND payload <> '{}'::jsonb
                        """,
                        (str(error_detail)[:2000], max(0.0, float(delay_sec)), job_id),
                    )
                    return bool(cur.rowcount)
        except Exception:
            logger.exception("PostgreSQL 延迟重试 ingest 任务失败: job_id=%s", job_id)
            return False

    @contextmanager
    def document_ingest_lock(
        self,
        document_id: str,
        *,
        settings: Optional[Settings] = None,
    ):
        settings = settings or get_settings()
        stable_id = str(document_id or "").strip()
        if not stable_id:
            raise ValueError("document_id is required for ingestion locking")
        if not self.is_enabled(settings):
            with _document_locks_guard:
                lock = _document_locks.setdefault(stable_id, threading.RLock())
            with lock:
                yield
            return
        if not self.ensure_schema(settings):
            raise RuntimeError("PostgreSQL is unavailable for document locking")
        with self._get_connection(settings) as conn:
            if conn is None:
                raise RuntimeError("PostgreSQL is unavailable for document locking")
            lock_key = "knowledge:" + stable_id
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                    (lock_key,),
                )
            try:
                yield
            finally:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                        (lock_key,),
                    )

    def is_document_ingest_superseded(
        self,
        document_id: str,
        job_id: str,
        *,
        settings: Optional[Settings] = None,
    ) -> Optional[bool]:
        settings = settings or get_settings()
        if not self.is_enabled(settings):
            return False
        if not self.ensure_schema(settings):
            return None
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM agent_knowledge_documents AS active_document
                            JOIN agent_ingest_jobs AS active_job
                              ON active_job.job_id = active_document.job_id
                            JOIN agent_ingest_jobs AS candidate_job
                              ON candidate_job.job_id = %s
                            WHERE active_document.document_id = %s
                              AND active_document.active = TRUE
                              AND (
                                  active_job.created_at > candidate_job.created_at
                                  OR (
                                      active_job.created_at = candidate_job.created_at
                                      AND active_job.job_id > candidate_job.job_id
                                  )
                              )
                        )
                        """,
                        (job_id, document_id),
                    )
                    row = cur.fetchone()
                    return bool(row and row[0])
        except Exception:
            logger.exception(
                "PostgreSQL 检查 ingest CAS 失败: document_id=%s job_id=%s",
                document_id,
                job_id,
            )
            return None

    def fail_unrecoverable_ingest_jobs(
        self,
        *,
        settings: Optional[Settings] = None,
    ) -> int:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return 0
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return 0
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE agent_ingest_jobs
                        SET status = 'failed',
                            error_detail = 'ingest_payload_unavailable_after_restart',
                            lease_expires_at = NULL,
                            updated_at = NOW()
                        WHERE status IN ('queued', 'processing')
                          AND payload = '{}'::jsonb
                        """
                    )
                    return max(0, int(cur.rowcount or 0))
        except Exception:
            logger.exception("PostgreSQL 标记不可恢复 ingest 任务失败")
            return 0

    def get_ingest_job(
        self,
        job_id: str,
        settings: Optional[Settings] = None,
    ) -> Optional[dict[str, Any]]:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return None
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT job_id, source, text_length, status, error_detail,
                               tenant_id, owner_id
                        FROM agent_ingest_jobs
                        WHERE job_id = %s
                        """,
                        (job_id,),
                    )
                    row = cur.fetchone()
            if row is None:
                return None
            return {
                "job_id": str(row[0]),
                "source": str(row[1]),
                "text_length": int(row[2]),
                "status": str(row[3]),
                "error_detail": str(row[4] or ""),
                "tenant_id": str(row[5] or ""),
                "owner_id": str(row[6] or ""),
            }
        except Exception:
            logger.exception("PostgreSQL 读取 ingest 任务失败: job_id=%s", job_id)
            return None

    def delete_session(
        self,
        session_id: str,
        *,
        target_generation: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> Optional[bool]:
        settings = settings or get_settings()
        if not self.is_enabled(settings):
            return False
        if not self.ensure_schema(settings):
            return None
        try:
            deleted = 0
            with self._get_connection(settings) as conn:
                if conn is None:
                    return None
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (session_id,),
                    )
                    cur.execute(
                        """
                        SELECT generation, active
                        FROM agent_session_tombstones
                        WHERE session_id = %s
                        FOR UPDATE
                        """,
                        (session_id,),
                    )
                    state_row = cur.fetchone()
                    current_generation = int(state_row[0] or 0) if state_row else 0
                    already_tombstoned = bool(state_row[1]) if state_row else False
                    requested_generation = (
                        max(0, int(target_generation))
                        if target_generation is not None
                        else None
                    )
                    if already_tombstoned:
                        next_generation = max(
                            current_generation,
                            requested_generation or current_generation,
                        )
                    else:
                        next_generation = max(
                            current_generation + 1,
                            requested_generation or 0,
                        )
                    cur.execute(
                        "DELETE FROM agent_chat_stream_events WHERE session_id = %s",
                        (session_id,),
                    )
                    deleted += max(0, int(cur.rowcount or 0))
                    cur.execute(
                        "DELETE FROM agent_chat_turns WHERE session_id = %s",
                        (session_id,),
                    )
                    deleted += max(0, int(cur.rowcount or 0))
                    cur.execute(
                        "DELETE FROM agent_sessions WHERE session_id = %s",
                        (session_id,),
                    )
                    deleted += max(0, int(cur.rowcount or 0))
                    cur.execute(
                        """
                        INSERT INTO agent_session_tombstones (
                            session_id, generation, active, deleted_at
                        )
                        VALUES (%s, %s, TRUE, NOW())
                        ON CONFLICT (session_id) DO UPDATE SET
                            generation = EXCLUDED.generation,
                            active = TRUE,
                            deleted_at = NOW()
                        """,
                        (session_id, next_generation),
                    )
                    cur.execute("COMMIT")
            return deleted > 0
        except Exception:
            logger.exception("PostgreSQL 删除 session 失败: session_id=%s", session_id)
            return None

    def is_session_tombstoned(
        self,
        session_id: str,
        *,
        settings: Optional[Settings] = None,
    ) -> Optional[bool]:
        state = self.get_session_state(session_id, settings=settings)
        return None if state is None else bool(state.get("tombstoned"))

    def clear_session_tombstone(
        self,
        session_id: str,
        *,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return not self.is_enabled(settings)
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (session_id,),
                    )
                    cur.execute(
                        """
                        UPDATE agent_session_tombstones
                        SET active = FALSE,
                            deleted_at = NOW()
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    cur.execute("COMMIT")
            return True
        except Exception:
            logger.exception("PostgreSQL 清除 session tombstone 失败: %s", session_id)
            return False

    def persist_knowledge_manifest(
        self,
        *,
        document_id: str,
        document_version: str,
        job_id: str,
        source: str,
        tenant_id: str,
        owner_id: str,
        visibility: str,
        allowed_roles: Iterable[str] = (),
        chunk_count: int,
        chunker_version: str,
        embedding_model: str,
        chunking_contract: Optional[dict[str, Any]] = None,
        settings: Optional[Settings] = None,
    ) -> bool:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return False
        try:
            acl = normalize_knowledge_acl(
                {
                    "tenant_id": tenant_id,
                    "owner_id": owner_id,
                    "visibility": visibility,
                    "allowed_roles": list(allowed_roles),
                }
            )
            tenant_id = str(acl["tenant_id"])
            owner_id = str(acl.get("owner_id") or "")
            visibility = str(acl["visibility"])
            allowed_roles_json = json.dumps(acl["allowed_roles"], ensure_ascii=False)
            contract_json = json.dumps(chunking_contract or {}, ensure_ascii=False)
            with self._get_connection(settings) as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("BEGIN")
                    if job_id:
                        cur.execute(
                            """
                            SELECT EXISTS (
                                SELECT 1
                                FROM agent_knowledge_documents AS active_document
                                JOIN agent_ingest_jobs AS active_job
                                  ON active_job.job_id = active_document.job_id
                                JOIN agent_ingest_jobs AS candidate_job
                                  ON candidate_job.job_id = %s
                                WHERE active_document.document_id = %s
                                  AND active_document.tenant_id = %s
                                  AND active_document.owner_id = %s
                                  AND active_document.active = TRUE
                                  AND (
                                      active_job.created_at > candidate_job.created_at
                                      OR (
                                          active_job.created_at = candidate_job.created_at
                                          AND active_job.job_id > candidate_job.job_id
                                      )
                                  )
                            )
                            """,
                            (job_id, document_id, tenant_id, owner_id),
                        )
                        if bool((cur.fetchone() or [False])[0]):
                            cur.execute("ROLLBACK")
                            logger.info(
                                "跳过已被新版本取代的 manifest: document_id=%s job_id=%s",
                                document_id,
                                job_id,
                            )
                            return False
                    cur.execute(
                        """
                        UPDATE agent_knowledge_documents
                        SET active = FALSE
                        WHERE document_id = %s
                          AND tenant_id = %s
                          AND owner_id = %s
                          AND document_version <> %s
                        """,
                        (document_id, tenant_id, owner_id, document_version),
                    )
                    cur.execute(
                        """
                        INSERT INTO agent_knowledge_documents (
                            document_id, document_version, job_id, source,
                            tenant_id, owner_id, visibility, allowed_roles, active, chunk_count,
                            chunker_version, chunking_contract, embedding_model
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s::jsonb, TRUE,
                            %s, %s, %s::jsonb, %s
                        )
                        ON CONFLICT (document_id, document_version) DO UPDATE SET
                            job_id = EXCLUDED.job_id,
                            source = EXCLUDED.source,
                            tenant_id = EXCLUDED.tenant_id,
                            owner_id = EXCLUDED.owner_id,
                            visibility = EXCLUDED.visibility,
                            allowed_roles = EXCLUDED.allowed_roles,
                            active = TRUE,
                            chunk_count = EXCLUDED.chunk_count,
                            chunker_version = EXCLUDED.chunker_version,
                            chunking_contract = EXCLUDED.chunking_contract,
                            embedding_model = EXCLUDED.embedding_model,
                            indexed_at = NOW()
                        WHERE agent_knowledge_documents.tenant_id = EXCLUDED.tenant_id
                          AND agent_knowledge_documents.owner_id = EXCLUDED.owner_id
                        """,
                        (
                            document_id,
                            document_version,
                            job_id,
                            source,
                            tenant_id,
                            owner_id,
                            visibility,
                            allowed_roles_json,
                            chunk_count,
                            chunker_version,
                            contract_json,
                            embedding_model,
                        ),
                    )
                    if cur.rowcount != 1:
                        cur.execute("ROLLBACK")
                        logger.error(
                            "拒绝跨租户或跨所有者覆盖 knowledge manifest: document_id=%s",
                            document_id,
                        )
                        return False
                    cur.execute("COMMIT")
            return True
        except Exception:
            logger.exception("PostgreSQL 持久化 knowledge manifest 失败")
            return False

    def list_knowledge_manifests(
        self,
        tenant_id: str,
        *,
        user_id: str = "",
        roles: Iterable[str] = (),
        is_admin: bool = False,
        limit: int = 100,
        settings: Optional[Settings] = None,
    ) -> list[dict[str, Any]]:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return []
        try:
            role_values = sorted({str(role) for role in roles if str(role)})
            with self._get_connection(settings) as conn:
                if conn is None:
                    return []
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT document_id, document_version, job_id, source,
                                 visibility, chunk_count, chunker_version,
                                 embedding_model, indexed_at, allowed_roles,
                                 owner_id, tenant_id, chunking_contract
                        FROM agent_knowledge_documents
                        WHERE tenant_id = %s
                          AND active = TRUE
                          AND (
                              %s
                              OR visibility = 'public'
                              OR (
                                  visibility = 'tenant'
                                  AND (
                                      allowed_roles = '[]'::jsonb
                                      OR allowed_roles ?| %s::text[]
                                  )
                              )
                              OR (
                                  visibility = 'private'
                                  AND owner_id = %s
                                  AND (
                                      allowed_roles = '[]'::jsonb
                                      OR allowed_roles ?| %s::text[]
                                  )
                              )
                          )
                        ORDER BY indexed_at DESC
                        LIMIT %s
                        """,
                        (
                            tenant_id,
                            bool(is_admin),
                            role_values,
                            user_id,
                            role_values,
                            max(1, min(limit, 500)),
                        ),
                    )
                    rows = cur.fetchall() or []
            access = RagAccessContext(
                tenant_id=tenant_id,
                user_id=user_id,
                roles=set(role_values),
                allow_legacy=False,
            )
            items = []
            for row in rows:
                allowed_role_values = _json_list(row[9])
                metadata = {
                    "visibility": str(row[4]),
                    "allowed_roles": allowed_role_values,
                    "owner_id": str(row[10] or ""),
                    "tenant_id": str(row[11] or ""),
                }
                if not is_admin and not document_is_authorized(metadata, access):
                    logger.warning(
                        "manifest SQL ACL 与应用 ACL 不一致，已 fail closed: document_id=%s",
                        row[0],
                    )
                    continue
                contract = row[12]
                if isinstance(contract, str):
                    try:
                        contract = json.loads(contract)
                    except json.JSONDecodeError:
                        contract = {}
                items.append(
                    {
                    "document_id": str(row[0]),
                    "document_version": str(row[1]),
                    "job_id": str(row[2]),
                    "source": str(row[3]),
                    "visibility": str(row[4]),
                    "chunk_count": int(row[5]),
                    "chunker_version": str(row[6]),
                    "embedding_model": str(row[7]),
                    "indexed_at": row[8].isoformat() if row[8] else "",
                    "allowed_roles": allowed_role_values,
                    "chunking_contract": dict(contract or {}),
                }
                )
            return items
        except Exception:
            logger.exception("PostgreSQL 读取 knowledge manifest 失败")
            return []


postgres_persistence = PostgresPersistence()


def _json_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted({str(item) for item in value if str(item)})
