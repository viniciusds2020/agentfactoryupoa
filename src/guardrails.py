"""Input sanitization and prompt injection protection."""
from __future__ import annotations

import re
import time
from collections import defaultdict

from src.utils import get_logger

logger = get_logger(__name__)

# ── Input limits ─────────────────────────────────────────────────────────────

MAX_QUESTION_LENGTH = 2000
MAX_HISTORY_MESSAGES = 20
MAX_COLLECTION_LENGTH = 64
COLLECTION_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+$")
ALLOWED_ROLES = {"user", "assistant"}

# ── Prompt injection patterns ────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r"ignor[ea]\s+(as\s+)?instru[çc][õo]es", re.IGNORECASE),
    re.compile(r"ignore\s+(previous|above|all)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(previous|above|all)", re.IGNORECASE),
    re.compile(r"forget\s+(your|all|previous)\s+(instructions|rules|prompts)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"agora\s+voc[eê]\s+[eé]\s+um", re.IGNORECASE),
    re.compile(r"novo\s+papel|new\s+role|act\s+as", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]|\<\|im_start\|\>", re.IGNORECASE),
    re.compile(r"repita\s+o\s+prompt|repeat\s+the\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"mostre\s+(seu|o)\s+prompt|show\s+(your|the)\s+prompt", re.IGNORECASE),
    re.compile(r"qual\s+[eé]\s+o\s+seu\s+prompt", re.IGNORECASE),
]


def detect_injection(text: str) -> str | None:
    """Return the matched pattern name if prompt injection is detected, else None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


# ── Input validation ─────────────────────────────────────────────────────────


def sanitize_question(question: str) -> str:
    """Validate and clean user question. Raises ValueError if invalid."""
    question = question.strip()
    if not question:
        raise ValueError("A pergunta não pode estar vazia.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"A pergunta excede o limite de {MAX_QUESTION_LENGTH} caracteres.")
    return question


def validate_collection(name: str) -> str:
    """Validate collection name format. Raises ValueError if invalid."""
    name = name.strip()
    if not name:
        raise ValueError("O nome da coleção não pode estar vazio.")
    if len(name) > MAX_COLLECTION_LENGTH:
        raise ValueError(f"O nome da coleção excede {MAX_COLLECTION_LENGTH} caracteres.")
    if not COLLECTION_PATTERN.match(name):
        raise ValueError("O nome da coleção deve conter apenas letras, números, _ e -.")
    return name


def sanitize_history(history: list) -> list:
    """Validate and truncate chat history. Removes invalid roles."""
    clean = []
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
        if role not in ALLOWED_ROLES:
            continue
        if not content or not isinstance(content, str):
            continue
        clean.append(msg)
    return clean


# ── Context sanitization (document injection defense) ────────────────────────

_CONTEXT_INJECTION_MARKERS = [
    re.compile(r"ignor[ea]\s+(as\s+)?instru[çc][õo]es[^.]*", re.IGNORECASE),
    re.compile(r"ignore\s+(previous|above|all)\s+instructions[^.]*", re.IGNORECASE),
    re.compile(r"system\s*:\s*[^\n]*", re.IGNORECASE),
    re.compile(r"<\s*system\s*>.*?<\s*/\s*system\s*>", re.IGNORECASE | re.DOTALL),
    re.compile(r"\[INST\].*?\[/INST\]", re.IGNORECASE | re.DOTALL),
]


def sanitize_context_chunk(text: str) -> str:
    """Neutralize prompt injection patterns found inside retrieved document chunks."""
    sanitized = text
    for pattern in _CONTEXT_INJECTION_MARKERS:
        sanitized = pattern.sub("[conteúdo removido]", sanitized)
    return sanitized


# ── Rate limiter (in-memory, per-IP) ─────────────────────────────────────────


class RateLimiter:
    """Simple sliding-window rate limiter. In-memory, single-process only.

    Limitations:
    - Resets on server restart.
    - Not shared across uvicorn workers (use ``--workers=1``).
    - For production with multiple workers/instances, replace with Redis-backed
      limiter using ZADD/ZRANGEBYSCORE, keeping the same ``is_allowed(key)`` API.
    - Alternatively, use rate limiting at the reverse proxy level (nginx/Traefik).
    """

    def __init__(self, max_requests: int = 30, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window
        hits = self._hits[key]
        self._hits[key] = [t for t in hits if t > cutoff]
        if len(self._hits[key]) >= self.max_requests:
            return False
        self._hits[key].append(now)
        return True


# Global rate limiters
chat_limiter = RateLimiter(max_requests=30, window_seconds=60)
ingest_limiter = RateLimiter(max_requests=10, window_seconds=60)
