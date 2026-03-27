"""Tabular extraction and chunking helpers."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_INTERNAL_FOOTER_RE = re.compile(r"classifica[cç][aã]o da informa[cç][aã]o:\s*interno", re.IGNORECASE)
_CODE_RE = re.compile(r"^\d{5,}$")


def _normalize_cell(value: object) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_column_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized or "coluna"


def _row_has_valid_code(row: list[str]) -> bool:
    for cell in row:
        if _CODE_RE.match(cell.strip()):
            return True
    return False


@dataclass
class TableRecord:
    row_index: int
    page_number: int | None
    fields: dict[str, str]
    texto_canonico: str
    raw_row: str


@dataclass
class TableExtractionResult:
    records: list[TableRecord]
    column_names: list[str]
    total_pages: int
    pages_with_tables: int


def build_texto_canonico(fields: dict, column_names: list[str]) -> str:
    parts: list[str] = []
    for col in column_names:
        val = (fields.get(col, "") or "").strip()
        if val:
            parts.append(f"{col}: {val}")
    return "; ".join(parts).strip()


def extract_tables_pdfplumber(pdf_path: str) -> TableExtractionResult:
    try:
        import pdfplumber
    except Exception:
        return TableExtractionResult(records=[], column_names=[], total_pages=0, pages_with_tables=0)

    records: list[TableRecord] = []
    header_norm: list[str] = []
    total_pages = 0
    pages_with_tables = 0
    row_idx = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            if any(t for t in tables if t):
                pages_with_tables += 1
            for table in tables:
                if not table:
                    continue
                raw_header = [_normalize_cell(cell) for cell in (table[0] or [])]
                if not any(raw_header):
                    continue
                local_header = [_normalize_column_name(col) for col in raw_header]
                if not header_norm:
                    header_norm = local_header
                for raw_row in table[1:]:
                    cells = [_normalize_cell(cell) for cell in (raw_row or [])]
                    if not any(cells):
                        continue
                    if _INTERNAL_FOOTER_RE.search(" ".join(cells)):
                        continue
                    if len(cells) >= len(local_header) and all(
                        cells[i].strip().lower() == raw_header[i].strip().lower() for i in range(min(len(raw_header), len(cells)))
                    ):
                        continue
                    if not _row_has_valid_code(cells):
                        continue
                    mapped = {}
                    for i, col in enumerate(local_header):
                        mapped[col] = cells[i] if i < len(cells) else ""
                    canon = build_texto_canonico(mapped, local_header)
                    if not canon:
                        continue
                    records.append(
                        TableRecord(
                            row_index=row_idx,
                            page_number=page_num,
                            fields=mapped,
                            texto_canonico=canon,
                            raw_row=" | ".join(cells),
                        )
                    )
                    row_idx += 1

    return TableExtractionResult(
        records=records,
        column_names=header_norm,
        total_pages=total_pages,
        pages_with_tables=pages_with_tables,
    )


def extract_tables_from_text(text: str) -> TableExtractionResult:
    lines = [ln.strip() for ln in (text or "").splitlines()]
    records: list[TableRecord] = []
    row_idx = 0
    col_names: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" not in line:
            i += 1
            continue
        if i + 1 >= len(lines) or "|" not in lines[i + 1]:
            i += 1
            continue
        raw_header = [seg.strip() for seg in line.strip("|").split("|")]
        sep = lines[i + 1]
        if not re.match(r"^\|?\s*:?-{2,}", sep):
            i += 1
            continue
        local_cols = [_normalize_column_name(c) for c in raw_header]
        if not col_names:
            col_names = local_cols
        i += 2
        while i < len(lines) and "|" in lines[i]:
            row_line = lines[i]
            cells = [seg.strip() for seg in row_line.strip("|").split("|")]
            if not any(cells):
                i += 1
                continue
            if _INTERNAL_FOOTER_RE.search(" ".join(cells)):
                i += 1
                continue
            if len(cells) >= len(raw_header) and all(
                cells[idx].strip().lower() == raw_header[idx].strip().lower()
                for idx in range(min(len(raw_header), len(cells)))
            ):
                i += 1
                continue
            if not _row_has_valid_code(cells):
                i += 1
                continue
            fields = {col: (cells[idx] if idx < len(cells) else "") for idx, col in enumerate(local_cols)}
            canon = build_texto_canonico(fields, local_cols)
            if canon:
                records.append(
                    TableRecord(
                        row_index=row_idx,
                        page_number=None,
                        fields=fields,
                        texto_canonico=canon,
                        raw_row=" | ".join(cells),
                    )
                )
                row_idx += 1
            i += 1
        continue

    return TableExtractionResult(records=records, column_names=col_names, total_pages=0, pages_with_tables=0)


def chunk_tabular_records(
    records: list[TableRecord],
    group_size: int = 5,
    max_chunk_chars: int = 512,
) -> list[tuple[str, dict]]:
    chunks: list[tuple[str, dict]] = []
    if not records:
        return chunks

    buf: list[TableRecord] = []
    buf_len = 0

    def _flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = "\n".join(rec.texto_canonico for rec in buf)
        metadata = {
            "chunk_type": "tabular",
            "row_start": buf[0].row_index,
            "row_end": buf[-1].row_index,
            "row_count": len(buf),
            "page_number": buf[0].page_number,
            "fields": buf[0].fields if len(buf) == 1 else {},
            "raw_rows": [rec.raw_row for rec in buf],
        }
        chunks.append((text, metadata))
        buf = []
        buf_len = 0

    for rec in records:
        rec_len = len(rec.texto_canonico)
        if rec_len > max_chunk_chars:
            _flush()
            chunks.append(
                (
                    rec.texto_canonico[:max_chunk_chars],
                    {
                        "chunk_type": "tabular",
                        "row_start": rec.row_index,
                        "row_end": rec.row_index,
                        "row_count": 1,
                        "page_number": rec.page_number,
                        "fields": rec.fields,
                        "raw_rows": [rec.raw_row],
                    },
                )
            )
            continue

        projected = buf_len + rec_len + (1 if buf else 0)
        if buf and (len(buf) >= group_size or projected > max_chunk_chars):
            _flush()

        buf.append(rec)
        buf_len = buf_len + rec_len + (1 if len(buf) > 1 else 0)

    _flush()
    return chunks
