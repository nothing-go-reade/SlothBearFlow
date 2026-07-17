from __future__ import annotations

import re


_PATTERNS = (
    (re.compile(r"(?<![\w.-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])", re.I), "[EMAIL_REDACTED]"),
    (re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"), "[PHONE_REDACTED]"),
    (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[ID_REDACTED]"),
    (re.compile(r"(?<!\d)\d{15}(?!\d)"), "[ID_REDACTED]"),
    (re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"), "[TOKEN_REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[TOKEN_REDACTED]"),
    (
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
        ),
        "[TOKEN_REDACTED]",
    ),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[TOKEN_REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"), "[TOKEN_REDACTED]"),
    (
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
        ),
        "[JWT_REDACTED]",
    ),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}={0,2}"), "Bearer [TOKEN_REDACTED]"),
    (re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/]{8,}={0,2}"), "Basic [TOKEN_REDACTED]"),
    (
        re.compile(
            r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s/@]+(@)"
        ),
        r"\1[SECRET_REDACTED]\2",
    ),
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|api[_-]?key|access[_-]?token|"
            r"refresh[_-]?token|client[_-]?secret|private[_-]?token|"
            r"session[_-]?(?:id|token)|cookie|secret)"
            r"[\"']?\s*[:=]\s*[\"']?([^\s,;\"']{6,})"
        ),
        r"\1=[SECRET_REDACTED]",
    ),
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
        ),
        "[PRIVATE_KEY_REDACTED]",
    ),
)

_PAYMENT_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def redact_memory_text(value: str, *, enabled: bool = True) -> str:
    text = str(value or "")
    if not enabled:
        return text
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = _PAYMENT_CARD.sub(_redact_payment_card, text)
    return text


def _redact_payment_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    if not 13 <= len(digits) <= 19 or not _passes_luhn(digits):
        return match.group(0)
    return "[PAYMENT_CARD_REDACTED]"


def _passes_luhn(digits: str) -> bool:
    checksum = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        checksum += value
    return checksum % 10 == 0
