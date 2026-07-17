from __future__ import annotations

import logging
import re
from typing import Any, List, Pattern

logger = logging.getLogger(__name__)

# 保守高精度正则：只命中明确的密钥/令牌形态，尽量不误伤正常 RAG 文本。
_PATTERNS: List[Pattern] = [
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),     # OpenAI 式密钥
    re.compile(r"AKIA[0-9A-Z]{16}"),                          # AWS Access Key ID
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/\-]{8,}={0,2}"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?"
        r"-----END [A-Z ]*PRIVATE KEY-----"
    ),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@[^\s]+"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|accesstoken|refresh[_-]?token|"
        r"refreshtoken|client[_-]?secret|clientsecret|private[_-]?key|privatekey|"
        r"token|password|secret)\s*[:=]\s*[^\s'\"]{6,}"
    ),                                                        # key: value 形态
]

_REDACTED = "[REDACTED]"


def scrub_observation(text: Any, settings: Any = None) -> str:
    """对工具输出脱敏后再回灌模型（应对敏感信息外泄 / OWASP LLM02）。

    gated by settings.tool_scrub_output（默认 True）。出错原样返回，绝不阻断。
    仅作用于工具「观测」，不触碰用户可见的最终回答。
    """
    original = str(text if text is not None else "")
    if settings is not None and not getattr(settings, "tool_scrub_output", True):
        return original
    try:
        result = original
        for pat in _PATTERNS:
            result = pat.sub(_REDACTED, result)
        return result
    except Exception:  # pragma: no cover
        logger.exception("scrub_observation failed")
        mode = str(getattr(settings, "tool_guard_mode", "enforce") or "enforce")
        return original if mode in {"off", "log"} else "[REDACTION_FAILED]"
