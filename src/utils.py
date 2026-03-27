"""Shared utilities: structured logging, request_id, deterministic chunk IDs."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_ENV = os.getenv("ENVIRONMENT", "development")

# ── PII masking (Brazilian formats) ──────────────────────────────────────────

_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"), "[CPF]"),
    (re.compile(r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}"), "[CNPJ]"),
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    (re.compile(r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}"), "[TELEFONE]"),
]


def mask_pii(text: str) -> str:
    """Replace common Brazilian PII patterns with placeholders."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_pii(record.getMessage()),
            "request_id": request_id_var.get(),
            "environment": _ENV,
        }
        if hasattr(record, "props") and isinstance(record.props, dict):
            masked_props = {k: mask_pii(str(v)) if isinstance(v, str) else v for k, v in record.props.items()}
            payload.update(masked_props)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def new_request_id() -> str:
    return str(uuid.uuid4())


def log_event(logger: logging.Logger, level: int, message: str, **props) -> None:
    logger.log(level, message, extra={"props": props})


def chunk_id(collection: str, doc_id: str, index: int) -> str:
    """Deterministic SHA-256 chunk ID — safe for re-indexing."""
    raw = f"{collection}:{doc_id}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()
