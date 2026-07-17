from __future__ import annotations

import logging
import hashlib
import re
import threading
from pathlib import Path
from typing import Any, Callable, List, Optional

from backend.src.slothbearflow_backend.learning.index_db import LearningIndex
from backend.src.slothbearflow_backend.learning.index_db import _parse_markdown
from backend.src.slothbearflow_backend.learning.schema import (
    MemoryItem,
    SkillItem,
    normalize_memory_type,
)
from backend.src.slothbearflow_backend.memory.privacy import redact_memory_text
from backend.src.slothbearflow_backend.rag.security import contains_prompt_injection

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_write_lock = threading.RLock()


def namespaced_learning_dir(base_dir: Any, tenant_id: str, user_id: str) -> Path:
    namespace = hashlib.sha256(f"{tenant_id}:{user_id}".encode()).hexdigest()[:24]
    return Path(base_dir) / "tenants" / namespace


def learning_dir_for(settings: Any, tenant_id: str, user_id: str) -> Path:
    is_local = (
        tenant_id == getattr(settings, "auth_local_tenant_id", "local")
        and user_id == getattr(settings, "auth_local_user_id", "local-user")
    )
    if not getattr(settings, "auth_required", False) and is_local:
        return Path(settings.review_base_dir)
    return namespaced_learning_dir(settings.review_base_dir, tenant_id, user_id)


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

    def upsert_memory(
        self,
        item: MemoryItem,
        *,
        write_guard: Optional[Callable[[], bool]] = None,
    ) -> Optional[Path]:
        if self._unsafe_item(item.description, item.body):
            logger.warning("拒绝疑似污染的 memory: %s", item.name)
            return None
        slug = self._safe_slug(item.name)
        path = self._resolve_within(self.memory_dir, slug)
        if path is None:
            return None
        type_ = normalize_memory_type(item.type)
        content = (
            "---\n"
            f"name: {self._esc(item.name) or slug}\n"
            f"description: {self._esc(redact_memory_text(item.description))}\n"
            "metadata:\n"
            f"  type: {type_}\n"
            f"  confidence: {item.confidence:.3f}\n"
            f"  source_tenant_id: {self._esc(item.source_tenant_id)}\n"
            f"  source_user_id: {self._esc(item.source_user_id)}\n"
            f"  source_session_id: {self._esc(item.source_session_id)}\n"
            f"  source_turn_id: {self._esc(item.source_turn_id)}\n"
            f"  source_generation: {max(0, int(item.source_generation))}\n"
            "---\n\n"
            f"{redact_memory_text(item.body).strip()}\n"
        )
        try:
            with _write_lock:
                if write_guard is not None and not write_guard():
                    logger.info("memory 写入时来源会话已失效: %s", item.name)
                    return None
                self._atomic_write(path, content)
                indexed = self.index.upsert(
                    kind="memory",
                    name=slug,
                    type_=type_,
                    description=self._esc(redact_memory_text(item.description)),
                    rel_path=f"memory/{path.name}",
                    mtime=path.stat().st_mtime,
                    body=redact_memory_text(item.body).strip(),
                    source_tenant_id=item.source_tenant_id,
                    source_user_id=item.source_user_id,
                    source_session_id=item.source_session_id,
                    source_turn_id=item.source_turn_id,
                    source_generation=item.source_generation,
                )
                if not indexed:
                    raise RuntimeError("memory index update failed")
        except Exception:
            logger.exception("写入 memory 失败: %s", path)
            return None
        logger.info("memory 已落盘: %s", path.name)
        return path

    def upsert_skill(
        self,
        item: SkillItem,
        *,
        write_guard: Optional[Callable[[], bool]] = None,
    ) -> Optional[Path]:
        if self._unsafe_item(item.trigger, item.body):
            logger.warning("拒绝疑似污染的 skill: %s", item.name)
            return None
        slug = self._safe_slug(item.name)
        path = self._resolve_within(self.skills_dir, slug)
        if path is None:
            return None
        content = (
            "---\n"
            f"name: {self._esc(item.name) or slug}\n"
            f"trigger: {self._esc(redact_memory_text(item.trigger))}\n"
            "metadata:\n"
            f"  confidence: {item.confidence:.3f}\n"
            f"  source_tenant_id: {self._esc(item.source_tenant_id)}\n"
            f"  source_user_id: {self._esc(item.source_user_id)}\n"
            f"  source_session_id: {self._esc(item.source_session_id)}\n"
            f"  source_turn_id: {self._esc(item.source_turn_id)}\n"
            f"  source_generation: {max(0, int(item.source_generation))}\n"
            "---\n\n"
            f"{redact_memory_text(item.body).strip()}\n"
        )
        try:
            with _write_lock:
                if write_guard is not None and not write_guard():
                    logger.info("skill 写入时来源会话已失效: %s", item.name)
                    return None
                self._atomic_write(path, content)
                indexed = self.index.upsert(
                    kind="skills",
                    name=slug,
                    trigger=self._esc(redact_memory_text(item.trigger)),
                    description=self._esc(redact_memory_text(item.trigger)),
                    rel_path=f"skills/{path.name}",
                    mtime=path.stat().st_mtime,
                    body=redact_memory_text(item.body).strip(),
                    source_tenant_id=item.source_tenant_id,
                    source_user_id=item.source_user_id,
                    source_session_id=item.source_session_id,
                    source_turn_id=item.source_turn_id,
                    source_generation=item.source_generation,
                )
                if not indexed:
                    raise RuntimeError("skill index update failed")
        except Exception:
            logger.exception("写入 skill 失败: %s", path)
            return None
        logger.info("skill 已落盘: %s", path.name)
        return path

    def save_many(
        self,
        memories: Optional[List[MemoryItem]] = None,
        skills: Optional[List[SkillItem]] = None,
        *,
        max_items: int = 5,
        source_tenant_id: Optional[str] = None,
        source_user_id: Optional[str] = None,
        source_session_id: Optional[str] = None,
        source_turn_id: Optional[str] = None,
        source_generation: Optional[int] = None,
        write_guard: Optional[Callable[[], bool]] = None,
    ) -> dict:
        written = {"memories": [], "skills": []}
        for item in (memories or [])[:max_items]:
            if not str(item.name or "").strip():
                continue
            item = self._with_source(
                item,
                tenant_id=source_tenant_id,
                user_id=source_user_id,
                session_id=source_session_id,
                turn_id=source_turn_id,
                generation=source_generation,
            )
            path = self.upsert_memory(item, write_guard=write_guard)
            if path is not None:
                written["memories"].append(path.name)
        for item in (skills or [])[:max_items]:
            if not str(item.name or "").strip():
                continue
            item = self._with_source(
                item,
                tenant_id=source_tenant_id,
                user_id=source_user_id,
                session_id=source_session_id,
                turn_id=source_turn_id,
                generation=source_generation,
            )
            path = self.upsert_skill(item, write_guard=write_guard)
            if path is not None:
                written["skills"].append(path.name)
        return written

    def delete_by_source(
        self,
        *,
        tenant_id: str,
        session_id: str,
        generation: int,
    ) -> dict[str, int]:
        expected_generation = max(0, int(generation))
        deleted = {"memories": 0, "skills": 0}
        with _write_lock:
            for kind, directory, result_key in (
                ("memory", self.memory_dir, "memories"),
                ("skills", self.skills_dir, "skills"),
            ):
                for path in sorted(directory.glob("*.md")):
                    metadata, _ = _parse_markdown(path)
                    if not self._source_matches(
                        metadata,
                        tenant_id=tenant_id,
                        session_id=session_id,
                        generation=expected_generation,
                    ):
                        continue
                    path.unlink()
                    if not self.index.delete(kind, path.stem):
                        raise RuntimeError(
                            f"learning index deletion failed for {kind}/{path.stem}"
                        )
                    deleted[result_key] += 1

            self.index.reindex_from_disk(self.base_dir)
            remaining = [
                row
                for row in self.index.all()
                if self._source_matches(
                    row,
                    tenant_id=tenant_id,
                    session_id=session_id,
                    generation=expected_generation,
                )
            ]
            if remaining:
                raise RuntimeError("learning cascade left stale index entries")
        return deleted

    @staticmethod
    def _with_source(
        item: Any,
        *,
        tenant_id: Optional[str],
        user_id: Optional[str],
        session_id: Optional[str],
        turn_id: Optional[str],
        generation: Optional[int],
    ) -> Any:
        if all(
            value is None
            for value in (tenant_id, user_id, session_id, turn_id, generation)
        ):
            return item
        return item.model_copy(
            update={
                "source_tenant_id": str(tenant_id or ""),
                "source_user_id": str(user_id or ""),
                "source_session_id": str(session_id or ""),
                "source_turn_id": str(turn_id or ""),
                "source_generation": max(0, int(generation or 0)),
            }
        )

    @staticmethod
    def _source_matches(
        metadata: dict[str, Any],
        *,
        tenant_id: str,
        session_id: str,
        generation: int,
    ) -> bool:
        try:
            source_generation = int(metadata.get("source_generation") or 0)
        except (TypeError, ValueError):
            return False
        return (
            str(metadata.get("source_tenant_id") or "") == str(tenant_id)
            and str(metadata.get("source_session_id") or "") == str(session_id)
            and source_generation == generation
        )

    @staticmethod
    def _unsafe_item(*values: str) -> bool:
        return any(contains_prompt_injection(str(value or "")) for value in values)

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with _write_lock:
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(path)

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
                if self._unsafe_item(str(row.get("name") or ""), desc, body):
                    logger.warning("读回时跳过疑似污染的 memory: %s", row.get("name"))
                    continue
                summary = desc or body[:120]
                lines.append(f"- {row.get('name')}: {summary}")
        if skills:
            lines.append("【可复用技巧】")
            for row in skills:
                trig = str(row.get("trigger") or "").strip()
                body = str(row.get("body") or "").strip()
                if self._unsafe_item(str(row.get("name") or ""), trig, body):
                    logger.warning("读回时跳过疑似污染的 skill: %s", row.get("name"))
                    continue
                summary = body[:160]
                prefix = f"（{trig}）" if trig else ""
                lines.append(f"- {row.get('name')}: {prefix}{summary}")
        block = "\n".join(lines).strip()
        if len(block) > budget_chars:
            block = block[:budget_chars].rstrip() + " …"
        return block
