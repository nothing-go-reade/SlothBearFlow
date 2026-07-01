from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, List, Optional

from backend.src.slothbearflow_backend.learning.index_db import LearningIndex
from backend.src.slothbearflow_backend.learning.schema import (
    MemoryItem,
    SkillItem,
    normalize_memory_type,
)

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


class LearningStore:
    """review 产出的唯一写入面（= Hermes 的 memory/skills 白名单在本项目的映射）。

    Markdown 文件为真相源，SQLite 为派生索引。所有写操作仅落在
    base_dir/{memory,skills} 下，路径穿越被拒绝。
    """

    def __init__(self, base_dir: Any) -> None:  # type: ignore[name-defined]
        self.base_dir = Path(base_dir).resolve()
        self.memory_dir = self.base_dir / "memory"
        self.skills_dir = self.base_dir / "skills"
        self._ensure_dirs()
        self.index = LearningIndex(self.base_dir / "index.sqlite")

    def _ensure_dirs(self) -> None:
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.skills_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("LearningStore 目录创建失败: %s", self.base_dir)

    def _safe_slug(self, name: str) -> str:
        slug = _SLUG_RE.sub("-", str(name or "").strip().lower()).strip("-")
        slug = slug[:80]
        return slug or "item"

    def _resolve_within(self, directory: Path, slug: str) -> Optional[Path]:
        candidate = (directory / f"{slug}.md").resolve()
        try:
            if not candidate.is_relative_to(directory.resolve()):
                logger.warning("拒绝越界写入: %s", candidate)
                return None
        except AttributeError:  # pragma: no cover - py<3.9
            if directory.resolve() not in candidate.parents:
                return None
        return candidate

    @staticmethod
    def _esc(value: str) -> str:
        return str(value or "").replace("\n", " ").strip()

    def upsert_memory(self, item: MemoryItem) -> Optional[Path]:
        slug = self._safe_slug(item.name)
        path = self._resolve_within(self.memory_dir, slug)
        if path is None:
            return None
        type_ = normalize_memory_type(item.type)
        content = (
            "---\n"
            f"name: {self._esc(item.name) or slug}\n"
            f"description: {self._esc(item.description)}\n"
            "metadata:\n"
            f"  type: {type_}\n"
            "---\n\n"
            f"{item.body.strip()}\n"
        )
        try:
            path.write_text(content, encoding="utf-8")
        except Exception:
            logger.exception("写入 memory 失败: %s", path)
            return None
        self.index.upsert(
            kind="memory",
            name=slug,
            type_=type_,
            description=self._esc(item.description),
            rel_path=f"memory/{path.name}",
            mtime=path.stat().st_mtime,
            body=item.body.strip(),
        )
        logger.info("memory 已落盘: %s", path.name)
        return path

    def upsert_skill(self, item: SkillItem) -> Optional[Path]:
        slug = self._safe_slug(item.name)
        path = self._resolve_within(self.skills_dir, slug)
        if path is None:
            return None
        content = (
            "---\n"
            f"name: {self._esc(item.name) or slug}\n"
            f"trigger: {self._esc(item.trigger)}\n"
            "---\n\n"
            f"{item.body.strip()}\n"
        )
        try:
            path.write_text(content, encoding="utf-8")
        except Exception:
            logger.exception("写入 skill 失败: %s", path)
            return None
        self.index.upsert(
            kind="skills",
            name=slug,
            trigger=self._esc(item.trigger),
            description=self._esc(item.trigger),
            rel_path=f"skills/{path.name}",
            mtime=path.stat().st_mtime,
            body=item.body.strip(),
        )
        logger.info("skill 已落盘: %s", path.name)
        return path

    def save_many(
        self,
        memories: Optional[List[MemoryItem]] = None,
        skills: Optional[List[SkillItem]] = None,
        *,
        max_items: int = 5,
    ) -> dict:
        written = {"memories": [], "skills": []}
        for item in (memories or [])[:max_items]:
            if not str(item.name or "").strip():
                continue
            path = self.upsert_memory(item)
            if path is not None:
                written["memories"].append(path.name)
        for item in (skills or [])[:max_items]:
            if not str(item.name or "").strip():
                continue
            path = self.upsert_skill(item)
            if path is not None:
                written["skills"].append(path.name)
        return written

    def select_for_injection(self, query: str, budget_chars: int) -> str:
        """读回：按相关度选条目，拼出有界的「长期记忆/技巧」注入块。

        剔除易变字段（mtime/路径），仅用稳定内容，以在学习集不变时保持
        system prompt 前缀字节稳定（保护 prefix cache）。
        """
        if budget_chars <= 0 or not self.index.ready:
            return ""
        memories = self.index.search(query, kind="memory", limit=5)
        skills = self.index.search(query, kind="skills", limit=5)
        if not memories and not skills:
            return ""
        lines: List[str] = []
        if memories:
            lines.append("【长期记忆（用户偏好/背景）】")
            for row in memories:
                desc = str(row.get("description") or "").strip()
                body = str(row.get("body") or "").strip()
                summary = desc or body[:120]
                lines.append(f"- {row.get('name')}: {summary}")
        if skills:
            lines.append("【可复用技巧】")
            for row in skills:
                trig = str(row.get("trigger") or "").strip()
                body = str(row.get("body") or "").strip()
                summary = body[:160]
                prefix = f"（{trig}）" if trig else ""
                lines.append(f"- {row.get('name')}: {prefix}{summary}")
        block = "\n".join(lines).strip()
        if len(block) > budget_chars:
            block = block[:budget_chars].rstrip() + " …"
        return block
