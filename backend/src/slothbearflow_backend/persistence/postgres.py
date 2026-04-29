from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional

from slothbearflow_backend import Settings, get_settings

logger = logging.getLogger(__name__)


class PostgresPersistence:
    def __init__(self) -> None:
        self._schema_ready = False
        self._driver_checked = False
        self._psycopg = None

    def is_enabled(self, settings: Optional[Settings] = None) -> bool:
        settings = settings or get_settings()
        return bool(
            settings.enable_postgres_persistence and str(settings.postgres_dsn).strip()
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
        return psycopg.connect(
            settings.postgres_dsn,
            connect_timeout=settings.postgres_connect_timeout,
            autocommit=True,
        )

    def ensure_schema(self, settings: Optional[Settings] = None) -> bool:
        settings = settings or get_settings()
        if not self.is_enabled(settings):
            return False
        if self._schema_ready:
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
                            session_id TEXT NOT NULL,
                            user_message TEXT NOT NULL,
                            assistant_message TEXT NOT NULL,
                            raw_output TEXT NOT NULL DEFAULT '',
                            response_source TEXT NOT NULL DEFAULT '',
                            tools_used JSONB NOT NULL DEFAULT '[]'::jsonb,
                            citations JSONB NOT NULL DEFAULT '[]'::jsonb,
                            response_mode TEXT NOT NULL DEFAULT 'invoke',
                            stream_format TEXT NOT NULL DEFAULT '',
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
                            seq INTEGER NOT NULL,
                            event_type TEXT NOT NULL,
                            content TEXT NOT NULL DEFAULT '',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
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
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
            self._schema_ready = True
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
            return {"enabled": True, "ready": True}
        except Exception as exc:  # noqa: BLE001
            return {"enabled": True, "ready": False, "reason": str(exc)}

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
        settings: Optional[Settings] = None,
    ) -> None:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return
        try:
            tools_json = json.dumps(list(tools_used), ensure_ascii=False)
            citations_json = json.dumps(list(citations), ensure_ascii=False)
            with self._get_connection(settings) as conn:
                if conn is None:
                    return
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_sessions (
                            session_id,
                            last_user_message,
                            last_assistant_message
                        )
                        VALUES (%s, %s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            last_user_message = EXCLUDED.last_user_message,
                            last_assistant_message = EXCLUDED.last_assistant_message,
                            updated_at = NOW()
                        """,
                        (session_id, user_message, assistant_message),
                    )
                    cur.execute(
                        """
                        INSERT INTO agent_chat_turns (
                            session_id,
                            user_message,
                            assistant_message,
                            raw_output,
                            response_source,
                            tools_used,
                            citations,
                            response_mode,
                            stream_format
                        )
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
                        """,
                        (
                            session_id,
                            user_message,
                            assistant_message,
                            raw_output,
                            response_source,
                            tools_json,
                            citations_json,
                            response_mode,
                            stream_format,
                        ),
                    )
        except Exception:
            logger.exception("PostgreSQL 持久化 chat turn 失败: session_id=%s", session_id)

    def persist_stream_events(
        self,
        *,
        session_id: str,
        events: Iterable[dict[str, Any]],
        settings: Optional[Settings] = None,
    ) -> None:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return
        rows = list(events)
        if not rows:
            return
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return
                with conn.cursor() as cur:
                    for idx, event in enumerate(rows):
                        cur.execute(
                            """
                            INSERT INTO agent_chat_stream_events (
                                session_id,
                                seq,
                                event_type,
                                content
                            )
                            VALUES (%s, %s, %s, %s)
                            """,
                            (
                                session_id,
                                int(event.get("seq", idx)),
                                str(event.get("event_type") or "chunk"),
                                str(event.get("content") or ""),
                            ),
                        )
        except Exception:
            logger.exception(
                "PostgreSQL 持久化 stream events 失败: session_id=%s", session_id
            )

    def load_session_snapshot(
        self,
        *,
        session_id: str,
        turn_limit: int,
        settings: Optional[Settings] = None,
    ) -> dict[str, Any]:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return {"messages": [], "summary": ""}
        if turn_limit <= 0:
            return {"messages": [], "summary": ""}
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return {"messages": [], "summary": ""}
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT summary
                        FROM agent_sessions
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
                    row = cur.fetchone()
                    summary = str(row[0]) if row and row[0] is not None else ""

                    cur.execute(
                        """
                        SELECT user_message, assistant_message
                        FROM agent_chat_turns
                        WHERE session_id = %s
                        ORDER BY created_at DESC, id DESC
                        LIMIT %s
                        """,
                        (session_id, turn_limit),
                    )
                    rows = cur.fetchall() or []

            rows = list(reversed(rows))
            messages: list[dict[str, str]] = []
            for user_message, assistant_message in rows:
                messages.append({"role": "user", "content": str(user_message or "")})
                messages.append(
                    {"role": "assistant", "content": str(assistant_message or "")}
                )
            return {"messages": messages, "summary": summary}
        except Exception:
            logger.exception("PostgreSQL 读取 session 快照失败: session_id=%s", session_id)
            return {"messages": [], "summary": ""}

    def persist_summary(
        self,
        session_id: str,
        summary: str,
        settings: Optional[Settings] = None,
    ) -> None:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_sessions (session_id, summary)
                        VALUES (%s, %s)
                        ON CONFLICT (session_id) DO UPDATE SET
                            summary = EXCLUDED.summary,
                            updated_at = NOW()
                        """,
                        (session_id, summary),
                    )
        except Exception:
            logger.exception("PostgreSQL 持久化摘要失败: session_id=%s", session_id)

    def persist_ingest_job(
        self,
        *,
        job_id: str,
        source: str,
        text_length: int,
        status: str,
        error_detail: str = "",
        settings: Optional[Settings] = None,
    ) -> None:
        settings = settings or get_settings()
        if not self.ensure_schema(settings):
            return
        try:
            with self._get_connection(settings) as conn:
                if conn is None:
                    return
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO agent_ingest_jobs (
                            job_id,
                            source,
                            text_length,
                            status,
                            error_detail
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (job_id) DO UPDATE SET
                            source = EXCLUDED.source,
                            text_length = EXCLUDED.text_length,
                            status = EXCLUDED.status,
                            error_detail = EXCLUDED.error_detail,
                            updated_at = NOW()
                        """,
                        (job_id, source, text_length, status, error_detail),
                    )
        except Exception:
            logger.exception("PostgreSQL 持久化 ingest 任务失败: job_id=%s", job_id)


postgres_persistence = PostgresPersistence()
