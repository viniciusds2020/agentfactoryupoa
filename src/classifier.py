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
    re.compile(r"^\s*CAP[IÃ]TULO\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*SE[CÃ‡][AÃƒ]O\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*T[IÃ]TULO\s+[IVXLCDM\d]+", re.IGNORECASE | re.MULTILINE),
]
_CATALOG_TOKENS = (
    "procedimento",
    "codigo",
    "descri",
    "cobertura",
    "autoriz",
    "prazo",
    "segmentacao",
    "ans",
)
_CODE_LINE_RE = re.compile(r"^\s*(?:[A-Z]{0,4}[-_/]?)?\d{4,}\b", re.IGNORECASE | re.MULTILINE)


@dataclass
class ClassificationResult:
    doc_type: DocType
    confidence: float
    table_ratio: float
    legal_pattern_count: int
    table_probe_rows: int = 0
    catalog_signal_count: int = 0
    code_line_count: int = 0


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


def _catalog_signal_count(text: str) -> int:
    haystack = (text or "").lower()
    return sum(1 for token in _CATALOG_TOKENS if token in haystack)


def _code_line_count(text: str) -> int:
    return len(_CODE_LINE_RE.findall(text or ""))


def _table_probe_from_pdf(pdf_path: str | None) -> int:
    if not pdf_path:
        return 0
    try:
        from src.tabular import extract_tables_pdfplumber

        extraction = extract_tables_pdfplumber(pdf_path)
        return len(extraction.records)
    except Exception as exc:
        log_event(logger, 30, "Tabular probe with pdfplumber extraction failed", pdf_path=pdf_path, error=str(exc))
        return 0


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

    table_probe_rows = _table_probe_from_pdf(pdf_path)
    catalog_signal_count = _catalog_signal_count(text)
    code_line_count = _code_line_count(text)

    if legal_detected and legal_pattern_count >= 3:
        return ClassificationResult(
            doc_type="legal",
            confidence=0.9,
            table_ratio=round(table_ratio, 4),
            legal_pattern_count=legal_pattern_count,
            table_probe_rows=table_probe_rows,
            catalog_signal_count=catalog_signal_count,
            code_line_count=code_line_count,
        )

    strong_tabular = (
        table_ratio > 0.4
        or table_probe_rows >= 8
        or (catalog_signal_count >= 3 and table_probe_rows >= 2)
        or code_line_count >= 8
    )
    mixed_tabular = (
        0.15 <= table_ratio <= 0.4
        or table_probe_rows >= 3
        or (catalog_signal_count >= 2 and code_line_count >= 3)
    )

    if strong_tabular:
        return ClassificationResult(
            doc_type="tabular",
            confidence=0.9 if table_ratio > 0.2 or table_probe_rows >= 8 else 0.82,
            table_ratio=round(table_ratio, 4),
            legal_pattern_count=legal_pattern_count,
            table_probe_rows=table_probe_rows,
            catalog_signal_count=catalog_signal_count,
            code_line_count=code_line_count,
        )
    if mixed_tabular:
        return ClassificationResult(
            doc_type="mixed",
            confidence=0.78,
            table_ratio=round(table_ratio, 4),
            legal_pattern_count=legal_pattern_count,
            table_probe_rows=table_probe_rows,
            catalog_signal_count=catalog_signal_count,
            code_line_count=code_line_count,
        )

    return ClassificationResult(
        doc_type="narrative",
        confidence=0.8,
        table_ratio=round(table_ratio, 4),
        legal_pattern_count=legal_pattern_count,
        table_probe_rows=table_probe_rows,
        catalog_signal_count=catalog_signal_count,
        code_line_count=code_line_count,
    )
