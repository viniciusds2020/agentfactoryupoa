from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.medical_catalog.normalize import dataframe_to_records, deduplicate_records, normalize_dataframe
from src.medical_catalog.schemas import CatalogIngestionConfig, ProcedureRecord
from src.medical_catalog.text import compact_whitespace, normalize_text


@dataclass
class ExtractionOutput:
    records: list[ProcedureRecord]
    extractor_chain: list[str]
    warnings: list[str]


def extract_records_from_pdf(pdf_path: str | Path, config: CatalogIngestionConfig) -> ExtractionOutput:
    path = Path(pdf_path)
    extractor_chain: list[str] = []
    warnings: list[str] = []
    records: list[ProcedureRecord] = []

    for extractor_name, extractor in (
        ("camelot", _extract_with_camelot),
        ("tabula", _extract_with_tabula),
        ("pymupdf_text", _extract_with_pymupdf_text),
        ("docling", _extract_with_docling),
    ):
        extractor_chain.append(extractor_name)
        try:
            records = extractor(path, config)
        except Exception as exc:
            warnings.append(f"{extractor_name}: {type(exc).__name__}: {exc}")
            continue
        if records:
            return ExtractionOutput(
                records=deduplicate_records(records),
                extractor_chain=extractor_chain,
                warnings=warnings,
            )
        warnings.append(f"{extractor_name}: nenhum registro aproveitável encontrado")

    return ExtractionOutput(records=[], extractor_chain=extractor_chain, warnings=warnings)


def _extract_with_camelot(pdf_path: Path, config: CatalogIngestionConfig) -> list[ProcedureRecord]:
    import camelot  # type: ignore

    found: list[ProcedureRecord] = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=config.pages, flavor=flavor)
        except Exception:
            continue
        for table in tables:
            df = normalize_dataframe(table.df, config)
            found.extend(
                dataframe_to_records(
                    df,
                    config=config,
                    source_file=pdf_path.name,
                    page_number=int(getattr(table, "page", 1) or 1),
                    extractor=f"camelot:{flavor}",
                )
            )
        if found:
            break
    return found


def _extract_with_tabula(pdf_path: Path, config: CatalogIngestionConfig) -> list[ProcedureRecord]:
    import tabula  # type: ignore

    tables = tabula.read_pdf(
        str(pdf_path),
        pages=config.pages,
        multiple_tables=True,
        guess=True,
        pandas_options={"dtype": str},
    )
    found: list[ProcedureRecord] = []
    for df in tables:
        normalized = normalize_dataframe(df, config)
        found.extend(
            dataframe_to_records(
                normalized,
                config=config,
                source_file=pdf_path.name,
                page_number=1,
                extractor="tabula",
            )
        )
    return found


def _extract_with_docling(pdf_path: Path, config: CatalogIngestionConfig) -> list[ProcedureRecord]:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    result = converter.convert(str(pdf_path))
    markdown = result.document.export_to_markdown(page_break_placeholder="<!-- PAGE_BREAK -->")
    records: list[ProcedureRecord] = []
    for page_number, page_content in enumerate(markdown.split("<!-- PAGE_BREAK -->"), start=1):
        for table_markdown in _extract_markdown_tables(page_content):
            df = pd.read_csv(pd.io.common.StringIO(table_markdown), sep="|", engine="python")
            normalized = normalize_dataframe(df, config)
            records.extend(
                dataframe_to_records(
                    normalized,
                    config=config,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    extractor="docling",
                )
            )
    return records


def _extract_markdown_tables(page_content: str) -> list[str]:
    tables: list[str] = []
    current: list[str] = []
    for line in page_content.splitlines():
        if "|" in line:
            current.append(line.strip())
        elif current:
            tables.append("\n".join(current))
            current = []
    if current:
        tables.append("\n".join(current))
    return tables


def _extract_with_pymupdf_text(pdf_path: Path, config: CatalogIngestionConfig) -> list[ProcedureRecord]:
    import fitz

    code_re = re.compile(config.code_pattern)
    records: list[ProcedureRecord] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            lines = [compact_whitespace(line) for line in text.splitlines() if compact_whitespace(line)]
            lines = _trim_catalog_lines(lines, config)
            if not lines:
                continue
            rows = _lines_to_rows(lines, config.expected_columns, code_re)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=config.expected_columns)
            normalized = normalize_dataframe(df, config)
            records.extend(
                dataframe_to_records(
                    normalized,
                    config=config,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    extractor="pymupdf_text",
                )
            )
    finally:
        doc.close()
    return records


def _trim_catalog_lines(lines: list[str], config: CatalogIngestionConfig) -> list[str]:
    trimmed = lines[:]
    if config.start_after:
        marker = normalize_text(config.start_after)
        for idx, line in enumerate(trimmed):
            if marker in normalize_text(line):
                trimmed = trimmed[idx + 1 :]
                break
    if config.stop_after:
        marker = normalize_text(config.stop_after)
        for idx, line in enumerate(trimmed):
            if marker in normalize_text(line):
                trimmed = trimmed[:idx]
                break

    normalized_headers = []
    for column in config.expected_columns:
        options = [column, *config.aliases.get(column, [])]
        normalized_headers.append({normalize_text(option) for option in options})
    header_indexes: list[int] = []
    expected_idx = 0
    for idx, line in enumerate(trimmed):
        if expected_idx < len(normalized_headers) and normalize_text(line) in normalized_headers[expected_idx]:
            header_indexes.append(idx)
            expected_idx += 1
            if expected_idx == len(normalized_headers):
                trimmed = trimmed[header_indexes[-1] + 1 :]
                break
    return trimmed


def _lines_to_rows(lines: list[str], columns: list[str], code_re: re.Pattern[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    pending: list[str] = []
    expected_size = len(columns)

    for line in lines:
        if code_re.match(line) and pending:
            row = _reshape_row(pending, expected_size)
            if row:
                rows.append(row)
            pending = [line]
            continue
        pending.append(line)

    if pending:
        row = _reshape_row(pending, expected_size)
        if row:
            rows.append(row)
    return rows


def _reshape_row(cells: list[str], expected_size: int) -> list[str] | None:
    if len(cells) < expected_size:
        return None
    if len(cells) == expected_size:
        return cells
    if expected_size < 3:
        return cells[:expected_size]

    middle_size = len(cells) - expected_size + 1
    row = [cells[0], " ".join(cells[1 : 1 + middle_size])]
    row.extend(cells[1 + middle_size :])
    return row[:expected_size]
