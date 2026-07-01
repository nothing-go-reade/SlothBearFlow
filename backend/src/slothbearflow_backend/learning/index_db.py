from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """与 rag 检索一致的轻量分词：ASCII 词 + 中文单字 + 中文二元组。"""
    text = str(text or "").lower()
    ascii_terms = re.findall(r"[a-z0-9_./:-]{2,}", text)
    cjk = re.findall(r"[一-鿿]", text)
    bigrams = ["".join(cjk[i : i + 2]) for i in range(max(0, len(cjk) - 1))]
    return ascii_terms + cjk + bigrams


class LearningIndex:
    """派生索引（SQLite）。Markdown 为真相源；本索引可由扫描 .md 完全重建。

    搜索用 Python 端关键词打分（语料小、可移植、零额外依赖），不依赖 FTS5。
    任何异常都降级为「索引不可用」，绝不影响主链路或复盘落盘。
    """

    def __init__(self, db_path: Any) -> None:
        self._db_path = str(db_path)
        self._ready = False
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entries (
                        kind TEXT NOT NULL,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL DEFAULT '',
                        description TEXT NOT NULL DEFAULT '',
                        trigger TEXT NOT NULL DEFAULT '',
                        rel_path TEXT NOT NULL DEFAULT '',
                        mtime REAL NOT NULL DEFAULT 0,
                        body TEXT NOT NULL DEFAULT '',
                        updated_at REAL NOT NULL DEFAULT 0,
                        PRIMARY KEY (kind, name)
                    )
                    """
                )
            self._ready = True
        except Exception:
            logger.exception("LearningIndex 初始化失败，索引降级关闭: %s", self._db_path)
            self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def upsert(
        self,
        *,
        kind: str,
        name: str,
        type_: str = "",
        description: str = "",
        trigger: str = "",
        rel_path: str = "",
        mtime: float = 0.0,
        body: str = "",
    ) -> None:
        if not self._ready:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO entries (
                        kind, name, type, description, trigger, rel_path, mtime, body, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(kind, name) DO UPDATE SET
                        type=excluded.type,
                        description=excluded.description,
                        trigger=excluded.trigger,
                        rel_path=excluded.rel_path,
                        mtime=excluded.mtime,
                        body=excluded.body,
                        updated_at=excluded.updated_at
                    """,
                    (
                        kind,
                        name,
                        type_,
                        description,
                        trigger,
                        rel_path,
                        float(mtime),
                        body,
                        float(mtime),
                    ),
                )
        except Exception:
            logger.exception("LearningIndex.upsert 失败: kind=%s name=%s", kind, name)

    def get(self, kind: str, name: str) -> Optional[Dict[str, Any]]:
        if not self._ready:
            return None
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM entries WHERE kind = ? AND name = ?",
                    (kind, name),
                ).fetchone()
            return dict(row) if row else None
        except Exception:
            logger.exception("LearningIndex.get 失败: kind=%s name=%s", kind, name)
            return None

    def all(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._ready:
            return []
        try:
            with self._connect() as conn:
                if kind:
                    rows = conn.execute(
                        "SELECT * FROM entries WHERE kind = ? ORDER BY updated_at DESC",
                        (kind,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM entries ORDER BY updated_at DESC"
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            logger.exception("LearningIndex.all 失败")
            return []

    def search(
        self, query: str, *, kind: Optional[str] = None, limit: int = 8
    ) -> List[Dict[str, Any]]:
        """关键词打分检索；query 为空时退化为按 recency 返回。"""
        rows = self.all(kind)
        if not rows:
            return []
        terms = set(_tokenize(query))
        if not terms:
            return rows[:limit]

        def score(row: Dict[str, Any]) -> tuple:
            haystack = " ".join(
                str(row.get(k) or "")
                for k in ("name", "description", "trigger", "body")
            )
            tokens = set(_tokenize(haystack))
            overlap = len(terms & tokens)
            return (overlap, row.get("updated_at") or 0.0)

        scored = sorted(rows, key=score, reverse=True)
        # 至少要有一个 term 命中才算相关；全 0 命中则不返回（避免读回噪声）。
        relevant = [r for r in scored if score(r)[0] > 0]
        return relevant[:limit]

    def delete(self, kind: str, name: str) -> None:
        if not self._ready:
            return
        try:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM entries WHERE kind = ? AND name = ?", (kind, name)
                )
        except Exception:
            logger.exception("LearningIndex.delete 失败: kind=%s name=%s", kind, name)

    def reindex_from_disk(self, base_dir: Any) -> int:
        """扫描 base_dir/{memory,skills}/*.md 全量重建索引；返回索引条数。"""
        if not self._ready:
            return 0
        base = Path(base_dir)
        count = 0
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM entries")
            for kind in ("memory", "skills"):
                sub = base / kind
                if not sub.is_dir():
                    continue
                for md in sorted(sub.glob("*.md")):
                    meta, body = _parse_markdown(md)
                    self.upsert(
                        kind=kind,
                        name=meta.get("name") or md.stem,
                        type_=meta.get("type", ""),
                        description=meta.get("description", ""),
                        trigger=meta.get("trigger", ""),
                        rel_path=f"{kind}/{md.name}",
                        mtime=md.stat().st_mtime,
                        body=body,
                    )
                    count += 1
        except Exception:
            logger.exception("LearningIndex.reindex_from_disk 失败")
        return count


def _parse_markdown(path: Path) -> tuple:
    """极简 frontmatter 解析：返回 (meta_dict, body)。无需 yaml 依赖。"""
    meta: Dict[str, str] = {}
    body = ""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return meta, body
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            header = text[3:end].strip("\n")
            body = text[end + 4 :].lstrip("\n")
            in_metadata = False
            for line in header.splitlines():
                if not line.strip():
                    continue
                if line.strip() == "metadata:":
                    in_metadata = True
                    continue
                if ":" in line:
                    key, _, value = line.strip().partition(":")
                    key = key.strip()
                    value = value.strip()
                    if in_metadata and key == "type":
                        meta["type"] = value
                    elif key in ("name", "description", "trigger", "type"):
                        meta[key] = value
                    if not line.startswith(" "):
                        in_metadata = key == "metadata"
            return meta, body
    return meta, text
