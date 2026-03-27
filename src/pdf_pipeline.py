"""PDF processing pipeline: extract → classify → clean → artifacts → quality gate.

Transforms raw PDF into structured blocks, generates clean .md and .json artifacts,
scores extraction quality, and gates indexing of low-quality parses.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.observability import metrics
from src.utils import get_logger, log_event

logger = get_logger(__name__)

# ── Block types ──────────────────────────────────────────────────────────────

BlockType = Literal[
    "body",
    "header",
    "footer",
    "table",
    "title",
    "section_header",
    "annex",
    "noise",
]

BLOCK_TYPES: set[str] = {
    "body", "header", "footer", "table", "title",
    "section_header", "annex", "noise",
}


@dataclass
class PDFBlock:
    """Intermediate representation of a single block extracted from a PDF page."""
    page_number: int
    block_id: str
    block_type: BlockType
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    source_parser: str = "pymupdf"
    confidence: float = 1.0
    section_hint: str = ""


@dataclass
class ParseQualityResult:
    """Quality assessment of a PDF parse."""
    score: float  # 0.0–1.0
    flags: list[str] = field(default_factory=list)
    noise_ratio: float = 0.0
    repetition_ratio: float = 0.0
    structure_detected: bool = False
    broken_line_ratio: float = 0.0
    empty_block_ratio: float = 0.0
    page_coherence: float = 1.0
    total_blocks: int = 0
    passed: bool = True


@dataclass
class PipelineResult:
    """Full result of the PDF processing pipeline."""
    blocks: list[PDFBlock]
    clean_text: str
    quality: ParseQualityResult
    md_path: str = ""
    json_path: str = ""
    status: str = "success"  # "success" | "needs_review" | "failed_parse"


# ── Extraction ───────────────────────────────────────────────────────────────

def _block_id(page_number: int, block_index: int, text: str) -> str:
    """Deterministic block ID from page, index and content prefix."""
    content = f"{page_number}:{block_index}:{text[:50]}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def extract_blocks(pdf_path: str | Path) -> list[PDFBlock]:
    """Extract structured blocks from PDF using PyMuPDF, preserving reading order.

    Each block carries its page number, bounding box, and raw text.
    Falls back to docling if PyMuPDF is not available.
    """
    pdf_path = Path(pdf_path)
    blocks: list[PDFBlock] = []

    try:
        import fitz  # PyMuPDF
    except ImportError:
        log_event(logger, 30, "PyMuPDF not available for block extraction", path=str(pdf_path))
        return _extract_blocks_docling(pdf_path)

    doc = fitz.open(str(pdf_path))

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

        for blk_idx, blk in enumerate(page_blocks):
            # Skip image blocks (type=1)
            if blk.get("type", 0) == 1:
                continue

            # Extract text from spans within lines
            text_parts: list[str] = []
            for line in blk.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                text_parts.append(line_text)
            text = "\n".join(text_parts).strip()

            if not text:
                continue

            bbox = (
                blk.get("bbox", (0, 0, 0, 0))[0],
                blk.get("bbox", (0, 0, 0, 0))[1],
                blk.get("bbox", (0, 0, 0, 0))[2],
                blk.get("bbox", (0, 0, 0, 0))[3],
            )

            bid = _block_id(page_num, blk_idx, text)
            blocks.append(PDFBlock(
                page_number=page_num,
                block_id=bid,
                block_type="body",  # initial classification
                text=text,
                bbox=bbox,
                source_parser="pymupdf",
            ))

    doc.close()
    log_event(logger, 20, "PDF blocks extracted", path=str(pdf_path), total_blocks=len(blocks))
    return blocks


def _extract_blocks_docling(pdf_path: Path) -> list[PDFBlock]:
    """Fallback extraction using docling when PyMuPDF is unavailable."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    pdf_opts.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )
    result = converter.convert(str(pdf_path))
    text = result.document.export_to_markdown(
        page_break_placeholder="<!-- PAGE_BREAK -->",
    )

    # Split by page breaks and create one block per paragraph per page
    pages = text.split("<!-- PAGE_BREAK -->")
    blocks: list[PDFBlock] = []
    for page_idx, page_text in enumerate(pages):
        page_num = page_idx + 1
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        for blk_idx, para in enumerate(paragraphs):
            bid = _block_id(page_num, blk_idx, para)
            blocks.append(PDFBlock(
                page_number=page_num,
                block_id=bid,
                block_type="body",
                text=para,
                bbox=(0, 0, 0, 0),
                source_parser="docling",
            ))

    log_event(logger, 20, "PDF blocks extracted via docling fallback",
              path=str(pdf_path), total_blocks=len(blocks))
    return blocks


# ── Classification ───────────────────────────────────────────────────────────

# Patterns for detecting block types
_TITLE_RE = re.compile(
    r"^(T[IÍ]TULO\s+[IVXLCDM\d]+|ESTATUTO\s+SOCIAL|REGULAMENTO|REGIMENTO)\b",
    re.IGNORECASE,
)
_SECTION_HEADER_RE = re.compile(
    r"^(CAP[IÍ]TULO\s+[IVXLCDM\d]+|SE[CÇ][AÃ]O\s+[IVXLCDM\d]+|PARTE\s+[IVXLCDM\d]+)",
    re.IGNORECASE,
)
_ANNEX_RE = re.compile(r"^ANEXO\s+[IVXLCDM\d]+", re.IGNORECASE)
_TABLE_INDICATORS = re.compile(r"(\|.*\|)|(\+[-=]+\+)", re.MULTILINE)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_FOOTER_PATTERNS = [
    re.compile(r"P[áa]gina\s+\d+\s+de\s+\d+", re.IGNORECASE),
    re.compile(r"Classifica[çc][ãa]o da Informa[çc][ãa]o", re.IGNORECASE),
    re.compile(r"\(Baixado por\s+", re.IGNORECASE),
]
_HEADER_PATTERNS = [
    re.compile(r"N[°º]/Rev\.?:", re.IGNORECASE),
    re.compile(r"Este documento faz parte", re.IGNORECASE),
]
_NOISE_PATTERNS = [
    re.compile(r"^<!--\s*image\s*-->$", re.IGNORECASE),
    re.compile(r"^\s*\*{3,}\s*$"),
    re.compile(r"^[-_=]{10,}$"),
]


def _classify_block_type(block: PDFBlock, page_height: float = 842.0) -> BlockType:
    """Classify a block based on content and position heuristics."""
    text = block.text.strip()
    _, y0, _, y1 = block.bbox

    # Noise detection
    if len(text) < 3:
        return "noise"
    for pattern in _NOISE_PATTERNS:
        if pattern.match(text):
            return "noise"

    # Page numbers (standalone digits near top/bottom)
    if _PAGE_NUMBER_RE.match(text):
        return "noise"

    # Header detection (top 12% of page)
    if page_height > 0 and y0 < page_height * 0.12:
        for pattern in _HEADER_PATTERNS:
            if pattern.search(text):
                return "header"

    # Footer detection (bottom 12% of page)
    if page_height > 0 and y0 > page_height * 0.88:
        for pattern in _FOOTER_PATTERNS:
            if pattern.search(text):
                return "footer"
        if _PAGE_NUMBER_RE.match(text):
            return "footer"

    # Structural headers
    if _TITLE_RE.match(text):
        return "title"
    if _SECTION_HEADER_RE.match(text):
        return "section_header"
    if _ANNEX_RE.match(text):
        return "annex"

    # Table detection
    if _TABLE_INDICATORS.search(text):
        return "table"

    # Footer/header patterns anywhere (for documents without bbox info)
    if block.bbox == (0, 0, 0, 0):
        for pattern in _FOOTER_PATTERNS:
            if pattern.search(text):
                return "footer"
        for pattern in _HEADER_PATTERNS:
            if pattern.search(text):
                return "header"

    return "body"


def classify_blocks(blocks: list[PDFBlock], page_height: float = 842.0) -> list[PDFBlock]:
    """Classify all blocks by type using heuristics. Mutates blocks in place."""
    for block in blocks:
        block.block_type = _classify_block_type(block, page_height)
    return blocks


# ── Repeated header/footer detection ────────────────────────────────────────

def _find_repeated_texts(blocks: list[PDFBlock], min_pages: int = 3) -> set[str]:
    """Find text that appears on multiple pages (likely headers/footers)."""
    # Normalize text for comparison
    text_pages: dict[str, set[int]] = {}
    for block in blocks:
        normalized = re.sub(r"\s+", " ", block.text.strip().lower())
        # Ignore very short texts (numbers, etc.)
        if len(normalized) < 10:
            continue
        text_pages.setdefault(normalized, set()).add(block.page_number)

    total_pages = len({b.page_number for b in blocks})
    threshold = min(min_pages, max(2, total_pages // 3))

    return {text for text, pages in text_pages.items() if len(pages) >= threshold}


def remove_repeated_headers_footers(blocks: list[PDFBlock]) -> list[PDFBlock]:
    """Mark blocks as header/footer/noise if their text repeats across pages."""
    repeated = _find_repeated_texts(blocks)
    if not repeated:
        return blocks

    for block in blocks:
        normalized = re.sub(r"\s+", " ", block.text.strip().lower())
        if normalized in repeated and block.block_type == "body":
            block.block_type = "header"  # repeated content = header/footer
            block.confidence = 0.8

    return blocks


# ── Paragraph recomposition ──────────────────────────────────────────────────

_BROKEN_LINE_RE = re.compile(r"[a-záàãâéêíóôõúüç,]\s*$", re.IGNORECASE)
_CONTINUATION_RE = re.compile(r"^[a-záàãâéêíóôõúüç]", re.IGNORECASE)


def recompose_paragraphs(blocks: list[PDFBlock]) -> list[PDFBlock]:
    """Merge blocks that are broken paragraphs (line ends mid-word/sentence)."""
    if not blocks:
        return blocks

    result: list[PDFBlock] = []
    buffer: PDFBlock | None = None

    for block in blocks:
        if block.block_type != "body":
            if buffer:
                result.append(buffer)
                buffer = None
            result.append(block)
            continue

        if buffer is None:
            buffer = PDFBlock(
                page_number=block.page_number,
                block_id=block.block_id,
                block_type="body",
                text=block.text,
                bbox=block.bbox,
                source_parser=block.source_parser,
                confidence=block.confidence,
                section_hint=block.section_hint,
            )
            continue

        # Check if this block is a continuation of the buffer
        buffer_text = buffer.text.rstrip()
        block_text = block.text.lstrip()

        if (
            block.page_number == buffer.page_number
            and _BROKEN_LINE_RE.search(buffer_text)
            and _CONTINUATION_RE.match(block_text)
        ):
            # Merge: continuation of broken paragraph
            buffer.text = buffer_text + " " + block_text
            # Expand bbox
            buffer.bbox = (
                min(buffer.bbox[0], block.bbox[0]),
                min(buffer.bbox[1], block.bbox[1]),
                max(buffer.bbox[2], block.bbox[2]),
                max(buffer.bbox[3], block.bbox[3]),
            )
        else:
            result.append(buffer)
            buffer = PDFBlock(
                page_number=block.page_number,
                block_id=block.block_id,
                block_type="body",
                text=block.text,
                bbox=block.bbox,
                source_parser=block.source_parser,
                confidence=block.confidence,
                section_hint=block.section_hint,
            )

    if buffer:
        result.append(buffer)

    return result


# ── Section hint assignment ──────────────────────────────────────────────────

def assign_section_hints(blocks: list[PDFBlock]) -> list[PDFBlock]:
    """Assign section_hint to body blocks based on preceding section_header/title blocks."""
    current_hint = ""
    for block in blocks:
        if block.block_type in ("title", "section_header"):
            current_hint = block.text.strip()
        if block.block_type == "body" and current_hint:
            block.section_hint = current_hint
    return blocks


# ── Clean text reconstruction ────────────────────────────────────────────────

_CONTENT_TYPES: set[str] = {"body", "table", "title", "section_header", "annex"}


def build_clean_text(blocks: list[PDFBlock]) -> str:
    """Reconstruct clean text from classified blocks, excluding noise/headers/footers."""
    parts: list[str] = []
    prev_page = 0

    for block in blocks:
        if block.block_type not in _CONTENT_TYPES:
            continue

        # Add page break marker between pages
        if block.page_number != prev_page and prev_page > 0:
            parts.append("\n<!-- PAGE_BREAK -->\n")
        prev_page = block.page_number

        # Add structural markers
        if block.block_type in ("title", "section_header"):
            parts.append(f"\n{block.text.strip()}\n")
        elif block.block_type == "annex":
            parts.append(f"\n{block.text.strip()}\n")
        elif block.block_type == "table":
            parts.append(f"\n{block.text.strip()}\n")
        else:
            parts.append(block.text.strip())

        parts.append("")  # blank line between blocks

    text = "\n".join(parts)
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ── Artifact generation ──────────────────────────────────────────────────────

def _ensure_processed_dir(base_dir: str = "data/processed") -> Path:
    """Ensure the processed artifacts directory exists."""
    path = Path(base_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _doc_id_from_path(pdf_path: str | Path) -> str:
    """Generate a stable doc_id from the PDF filename."""
    return Path(pdf_path).stem


def generate_markdown(blocks: list[PDFBlock], doc_id: str) -> str:
    """Generate a clean markdown representation of the document."""
    lines: list[str] = [f"# {doc_id}\n"]
    current_page = 0

    for block in blocks:
        if block.block_type not in _CONTENT_TYPES:
            continue

        if block.page_number != current_page:
            if current_page > 0:
                lines.append(f"\n---\n*Página {block.page_number}*\n")
            current_page = block.page_number

        if block.block_type == "title":
            lines.append(f"## {block.text.strip()}\n")
        elif block.block_type == "section_header":
            lines.append(f"### {block.text.strip()}\n")
        elif block.block_type == "annex":
            lines.append(f"#### {block.text.strip()}\n")
        elif block.block_type == "table":
            lines.append(f"\n{block.text.strip()}\n")
        else:
            lines.append(f"{block.text.strip()}\n")

    return "\n".join(lines)


def generate_json(blocks: list[PDFBlock], doc_id: str, quality: ParseQualityResult) -> dict:
    """Generate a structured JSON representation of the document."""
    return {
        "doc_id": doc_id,
        "total_blocks": len(blocks),
        "quality": asdict(quality),
        "blocks": [
            {
                "page_number": b.page_number,
                "block_id": b.block_id,
                "block_type": b.block_type,
                "text": b.text,
                "bbox": list(b.bbox),
                "source_parser": b.source_parser,
                "confidence": b.confidence,
                "section_hint": b.section_hint,
            }
            for b in blocks
        ],
    }


def save_artifacts(
    blocks: list[PDFBlock],
    doc_id: str,
    quality: ParseQualityResult,
    base_dir: str = "data/processed",
) -> tuple[str, str]:
    """Save .md and .json artifacts to data/processed/. Returns (md_path, json_path)."""
    out_dir = _ensure_processed_dir(base_dir)

    md_content = generate_markdown(blocks, doc_id)
    md_path = out_dir / f"{doc_id}.md"
    md_path.write_text(md_content, encoding="utf-8")

    json_content = generate_json(blocks, doc_id, quality)
    json_path = out_dir / f"{doc_id}.json"
    json_path.write_text(json.dumps(json_content, ensure_ascii=False, indent=2), encoding="utf-8")

    log_event(logger, 20, "Artifacts saved", doc_id=doc_id,
              md_path=str(md_path), json_path=str(json_path))
    return str(md_path), str(json_path)


# ── Quality scoring ──────────────────────────────────────────────────────────

def compute_quality_score(blocks: list[PDFBlock]) -> ParseQualityResult:
    """Compute a composite quality score for the PDF parse.

    Score components:
    - noise_ratio: % of blocks classified as noise
    - repetition_ratio: % of repeated text across pages
    - structure_detected: presence of titles/sections
    - broken_line_ratio: % of body blocks with suspiciously short lines
    - empty_block_ratio: % of empty/near-empty blocks
    - page_coherence: consistency of block count across pages
    """
    if not blocks:
        return ParseQualityResult(score=0.0, flags=["no_blocks"], passed=False, total_blocks=0)

    total = len(blocks)
    flags: list[str] = []

    # Noise ratio
    noise_count = sum(1 for b in blocks if b.block_type == "noise")
    noise_ratio = noise_count / total

    # Header/footer ratio (also non-content)
    hf_count = sum(1 for b in blocks if b.block_type in ("header", "footer"))

    # Empty block ratio
    empty_count = sum(1 for b in blocks if len(b.text.strip()) < 5)
    empty_ratio = empty_count / total

    # Structure detection
    has_structure = any(b.block_type in ("title", "section_header") for b in blocks)

    # Broken line ratio: body blocks with very short text (< 20 chars) that aren't structural
    body_blocks = [b for b in blocks if b.block_type == "body"]
    broken_count = sum(1 for b in body_blocks if len(b.text.strip()) < 20) if body_blocks else 0
    broken_ratio = broken_count / len(body_blocks) if body_blocks else 0.0

    # Repetition ratio
    repeated = _find_repeated_texts(blocks, min_pages=2)
    repeated_blocks = 0
    for b in blocks:
        normalized = re.sub(r"\s+", " ", b.text.strip().lower())
        if normalized in repeated:
            repeated_blocks += 1
    repetition_ratio = repeated_blocks / total if total > 0 else 0.0

    # Page coherence: std deviation of blocks per page (lower = more coherent)
    pages = {}
    for b in blocks:
        pages.setdefault(b.page_number, 0)
        pages[b.page_number] += 1

    if len(pages) > 1:
        counts = list(pages.values())
        mean_count = statistics.mean(counts)
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0
        page_coherence = max(0.0, 1.0 - (std_count / (mean_count + 1)))
    else:
        page_coherence = 1.0

    # Build flags
    if noise_ratio > 0.3:
        flags.append("high_noise")
    if repetition_ratio > 0.2:
        flags.append("high_repetition")
    if not has_structure:
        flags.append("no_structure")
    if broken_ratio > 0.4:
        flags.append("many_broken_lines")
    if empty_ratio > 0.3:
        flags.append("many_empty_blocks")
    if page_coherence < 0.5:
        flags.append("inconsistent_pages")

    # Composite score (weighted)
    score = (
        (1.0 - noise_ratio) * 0.25
        + (1.0 - repetition_ratio) * 0.15
        + (1.0 if has_structure else 0.3) * 0.20
        + (1.0 - broken_ratio) * 0.15
        + (1.0 - empty_ratio) * 0.10
        + page_coherence * 0.15
    )
    score = round(max(0.0, min(1.0, score)), 3)

    return ParseQualityResult(
        score=score,
        flags=flags,
        noise_ratio=round(noise_ratio, 3),
        repetition_ratio=round(repetition_ratio, 3),
        structure_detected=has_structure,
        broken_line_ratio=round(broken_ratio, 3),
        empty_block_ratio=round(empty_ratio, 3),
        page_coherence=round(page_coherence, 3),
        total_blocks=total,
        passed=True,  # set by gate
    )


# ── Validation gate ──────────────────────────────────────────────────────────

def quality_gate(quality: ParseQualityResult, min_score: float = 0.4) -> str:
    """Decide whether to proceed with indexing based on quality score.

    Returns:
        "success" — OK to index
        "needs_review" — borderline, index but flag
        "failed_parse" — do not index
    """
    if quality.score < min_score:
        quality.passed = False
        return "failed_parse"

    if quality.score < min_score + 0.15 or quality.flags:
        quality.passed = True
        return "needs_review"

    quality.passed = True
    return "success"


# ── Full pipeline ────────────────────────────────────────────────────────────

def process_pdf(
    pdf_path: str | Path,
    doc_id: str | None = None,
    save_artifacts_flag: bool = True,
    artifacts_dir: str = "data/processed",
    min_quality_score: float = 0.4,
) -> PipelineResult:
    """Run the full PDF processing pipeline.

    Steps:
    1. Extract blocks from PDF (PyMuPDF or docling fallback)
    2. Classify blocks by type (header, footer, body, title, table, noise, etc.)
    3. Remove repeated headers/footers across pages
    4. Recompose broken paragraphs
    5. Assign section hints
    6. Build clean text
    7. Score quality
    8. Apply validation gate
    9. Save artifacts (.md, .json)

    Returns PipelineResult with blocks, clean text, quality info, and artifact paths.
    """
    pdf_path = Path(pdf_path)
    doc_id = doc_id or _doc_id_from_path(pdf_path)

    # Step 1: Extract
    with metrics.time_block("pdf.pipeline.extract"):
        blocks = extract_blocks(pdf_path)

    if not blocks:
        quality = ParseQualityResult(
            score=0.0, flags=["no_blocks_extracted"], passed=False, total_blocks=0,
        )
        log_event(logger, 30, "PDF extraction produced no blocks",
                  doc_id=doc_id, path=str(pdf_path))
        return PipelineResult(
            blocks=[], clean_text="", quality=quality, status="failed_parse",
        )

    # Step 2: Classify
    with metrics.time_block("pdf.pipeline.classify"):
        blocks = classify_blocks(blocks)

    # Step 3: Remove repeated headers/footers
    with metrics.time_block("pdf.pipeline.dedup_headers"):
        blocks = remove_repeated_headers_footers(blocks)

    # Step 4: Recompose broken paragraphs
    with metrics.time_block("pdf.pipeline.recompose"):
        blocks = recompose_paragraphs(blocks)

    # Step 5: Assign section hints
    blocks = assign_section_hints(blocks)

    # Step 6: Build clean text
    with metrics.time_block("pdf.pipeline.build_text"):
        clean_text = build_clean_text(blocks)

    # Step 7: Quality score
    quality = compute_quality_score(blocks)

    # Step 8: Validation gate
    status = quality_gate(quality, min_score=min_quality_score)

    # Step 9: Artifacts
    md_path = ""
    json_path = ""
    if save_artifacts_flag:
        with metrics.time_block("pdf.pipeline.save_artifacts"):
            md_path, json_path = save_artifacts(blocks, doc_id, quality, base_dir=artifacts_dir)

    # Observability metrics
    metrics.increment("pdf.parse.total")
    if status == "success":
        metrics.increment("pdf.parse.success")
    elif status == "needs_review":
        metrics.increment("pdf.parse.needs_review")
    else:
        metrics.increment("pdf.parse.rejected")

    content_blocks = [b for b in blocks if b.block_type in _CONTENT_TYPES]
    if blocks:
        noise_blocks = sum(1 for b in blocks if b.block_type == "noise")
        table_blocks = sum(1 for b in blocks if b.block_type == "table")
        # Use observe for ratio metrics
        metrics.observe("pdf.blocks.noise_ratio", quality.noise_ratio * 100)
        if table_blocks:
            metrics.observe("pdf.blocks.table_ratio", (table_blocks / len(blocks)) * 100)
        metrics.observe("pdf.parse.quality", quality.score * 100)

    log_event(
        logger, 20, "PDF pipeline completed",
        doc_id=doc_id,
        path=str(pdf_path),
        total_blocks=len(blocks),
        content_blocks=len(content_blocks),
        quality_score=quality.score,
        quality_flags=quality.flags,
        status=status,
    )

    return PipelineResult(
        blocks=blocks,
        clean_text=clean_text,
        quality=quality,
        md_path=md_path,
        json_path=json_path,
        status=status,
    )
