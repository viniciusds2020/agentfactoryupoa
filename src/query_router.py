"""Heuristic query intent router for structured/vector retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Route = Literal["structured", "vector", "hybrid"]

_CODE_RE = re.compile(r"\b(\d{8}|\d{5,})\b")
_CODIGO_HINT_RE = re.compile(r"\bc[oó]digo\s+(\d{5,})\b", re.IGNORECASE)
_EMERGENCIA_RE = re.compile(r"\bemerg[eê]ncia\s+(sim|n[aã]o)\b", re.IGNORECASE)
_COBERTURA_RE = re.compile(r"\bcobertura\s+([a-z0-9_\- ]{2,60})\b", re.IGNORECASE)


@dataclass
class QueryIntent:
    route: Route
    structured_filters: dict[str, str]
    confidence: float


def detect_query_intent(question: str, collection_has_structured: bool) -> QueryIntent:
    if not collection_has_structured:
        return QueryIntent(route="vector", structured_filters={}, confidence=1.0)

    q = (question or "").strip()

    codigo = _CODIGO_HINT_RE.search(q)
    if codigo:
        return QueryIntent(
            route="structured",
            structured_filters={"codigo": codigo.group(1)},
            confidence=0.95,
        )

    generic_code = _CODE_RE.search(q)
    if generic_code:
        return QueryIntent(
            route="structured",
            structured_filters={"codigo": generic_code.group(1)},
            confidence=0.9,
        )

    filters: dict[str, str] = {}
    emerg = _EMERGENCIA_RE.search(q)
    if emerg:
        filters["emergencia"] = emerg.group(1).lower()
    cob = _COBERTURA_RE.search(q)
    if cob:
        filters["cobertura"] = cob.group(1).strip().lower()

    if filters:
        return QueryIntent(route="hybrid", structured_filters=filters, confidence=0.8)

    return QueryIntent(route="vector", structured_filters={}, confidence=0.75)
