"""Document classifier used by ingestion routing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from src.utils import get_logger, log_event

logger = get_logger(__name__)

DocType = Literal["tabular", "legal", "narrative", "mixed"]

_LEGAL_PATTERNS = [
    re.compile(r"^\s*Art\.?\s*\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*CAP[IÍ]TULO\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*SE[CÇ][AÃ]O\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*T[IÍ]TULO\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class ClassificationResult:
    doc_type: DocType
    confidence: float
    table_ratio: float
    legal_pattern_count: int


def _count_legal_patterns(text: str) -> int:
    return sum(len(p.findall(text or "")) for p in _LEGAL_PATTERNS)


def _table_ratio_from_blocks(blocks: list | None) -> float:
    if not blocks:
        return 0.0
    content = [b for b in blocks if getattr(b, "block_type", "") in {"body", "table", "title", "section_header", "annex"}]
    if not content:
        return 0.0
    table_count = sum(1 for b in content if getattr(b, "block_type", "") == "table")
    return table_count / len(content)


def _table_ratio_from_pdf(pdf_path: str | None) -> float:
    if not pdf_path:
        return 0.0
    try:
        import pdfplumber
    except Exception:
        return 0.0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return 0.0
            pages_with_tables = 0
            for page in pdf.pages:
                tables = page.extract_tables() or []
                if any(t for t in tables if t):
                    pages_with_tables += 1
            return pages_with_tables / total_pages
    except Exception as exc:
        log_event(logger, 30, "Table ratio probing with pdfplumber failed", pdf_path=pdf_path, error=str(exc))
        return 0.0


def classify_document(
    text: str,
    blocks: list | None = None,
    pdf_path: str | None = None,
) -> ClassificationResult:
    """Classify document type for ingestion routing."""
    from src.ingestion import _is_legal_document

    legal_pattern_count = _count_legal_patterns(text)
    legal_detected = _is_legal_document(text)
    table_ratio = _table_ratio_from_blocks(blocks)

    if table_ratio == 0.0 and pdf_path:
        table_ratio = _table_ratio_from_pdf(pdf_path)

    if legal_detected and legal_pattern_count >= 3:
        return ClassificationResult(
            doc_type="legal",
            confidence=0.9,
            table_ratio=round(table_ratio, 4),
            legal_pattern_count=legal_pattern_count,
        )

    if table_ratio > 0.4:
        return ClassificationResult(
            doc_type="tabular",
            confidence=0.9,
            table_ratio=round(table_ratio, 4),
            legal_pattern_count=legal_pattern_count,
        )
    if 0.2 <= table_ratio <= 0.4:
        return ClassificationResult(
            doc_type="mixed",
            confidence=0.75,
            table_ratio=round(table_ratio, 4),
            legal_pattern_count=legal_pattern_count,
        )

    return ClassificationResult(
        doc_type="narrative",
        confidence=0.8,
        table_ratio=round(table_ratio, 4),
        legal_pattern_count=legal_pattern_count,
    )
