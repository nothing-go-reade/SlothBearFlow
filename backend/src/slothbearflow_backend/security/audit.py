from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
import uuid
import fcntl
from pathlib import Path
from typing import Any, Dict, List, Optional


_lock = threading.Lock()


def audit_event(
    settings: Any,
    event_type: str,
    *,
    actor: str = "system",
    tenant_id: str = "local",
    target: str = "",
    outcome: str = "success",
    metadata: Dict[str, Any] = None,
) -> None:
    if not bool(getattr(settings, "audit_enabled", True)):
        return
    path = Path(str(getattr(settings, "audit_log_file", "logs/audit.jsonl"))).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_metadata = _safe_metadata(metadata or {})
    with _lock:
        descriptor = os.open(
            str(path),
            os.O_APPEND | os.O_CREAT | os.O_RDWR,
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            previous_hash = _read_last_hash(descriptor)
            event = {
                "event_id": uuid.uuid4().hex,
                "timestamp": time.time(),
                "event_type": str(event_type),
                "actor": str(actor)[:256],
                "tenant_id": str(tenant_id)[:128],
                "target": str(target)[:256],
                "outcome": str(outcome)[:64],
                "metadata": safe_metadata,
                "previous_hash": previous_hash,
            }
            canonical = json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            event["event_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
            os.write(
                descriptor,
                (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode(),
            )
            os.fsync(descriptor)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def read_recent_audit_events(settings: Any, limit: int = 100) -> List[Dict[str, Any]]:
    path = Path(str(getattr(settings, "audit_log_file", "logs/audit.jsonl"))).resolve()
    if not path.exists():
        return []
    rows = []
    for line in _read_locked_lines(path)[-max(1, min(limit, 500)) :]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except json.JSONDecodeError:
            continue
    return rows[::-1]


def verify_audit_chain(settings: Any, limit: int = 5000) -> Dict[str, Any]:
    path = Path(str(getattr(settings, "audit_log_file", "logs/audit.jsonl"))).resolve()
    if not path.exists():
        return {"valid": True, "checked": 0, "reason": "empty"}
    lines = _read_locked_lines(path)[-max(1, min(limit, 50000)) :]
    previous_hash: Optional[str] = None
    checked = 0
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return {"valid": False, "checked": checked, "reason": "invalid_json"}
        if not isinstance(event, dict):
            return {"valid": False, "checked": checked, "reason": "invalid_event"}
        event_hash = str(event.pop("event_hash", ""))
        canonical = json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        calculated = hashlib.sha256(canonical.encode()).hexdigest()
        if not event_hash or not hmac.compare_digest(event_hash, calculated):
            return {"valid": False, "checked": checked, "reason": "hash_mismatch"}
        if previous_hash is not None and event.get("previous_hash") != previous_hash:
            return {"valid": False, "checked": checked, "reason": "chain_break"}
        previous_hash = event_hash
        checked += 1
    return {"valid": True, "checked": checked, "reason": "ok"}


def _read_last_hash(descriptor: int) -> str:
    max_scan_bytes = 1024 * 1024
    try:
        end = os.lseek(descriptor, 0, os.SEEK_END)
        if end <= 0:
            return ""
        position = end
        buffer = b""
        while position > 0 and len(buffer) < max_scan_bytes:
            size = min(8192, position, max_scan_bytes - len(buffer))
            position -= size
            os.lseek(descriptor, position, os.SEEK_SET)
            buffer = os.read(descriptor, size) + buffer
            lines = buffer.splitlines()
            candidates = lines if position == 0 else lines[1:]
            for line in reversed(candidates):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(event, dict) and event.get("event_hash"):
                    return str(event["event_hash"])
        return ""
    except Exception:
        return ""


def _read_locked_lines(path: Path) -> List[str]:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except FileNotFoundError:
        return []
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        return _read_descriptor(descriptor).decode("utf-8").splitlines()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    blocked = {"password", "token", "secret", "authorization", "cookie", "content"}
    output = {}
    for key, value in metadata.items():
        if any(item in str(key).lower() for item in blocked):
            output[str(key)] = "[REDACTED]"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[str(key)] = value if not isinstance(value, str) else value[:500]
        elif isinstance(value, (list, tuple, set)):
            output[str(key)] = [str(item)[:200] for item in list(value)[:50]]
        else:
            output[str(key)] = str(value)[:500]
    return output
