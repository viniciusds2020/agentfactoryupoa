"""Tests for src.pdf_pipeline — block extraction, classification, cleaning, quality scoring."""
import json
from pathlib import Path

import pytest

from src.pdf_pipeline import (
    PDFBlock,
    ParseQualityResult,
    PipelineResult,
    _block_id,
    _classify_block_type,
    assign_section_hints,
    build_clean_text,
    classify_blocks,
    compute_quality_score,
    generate_json,
    generate_markdown,
    quality_gate,
    recompose_paragraphs,
    remove_repeated_headers_footers,
    save_artifacts,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_block(
    text: str,
    page: int = 1,
    block_type: str = "body",
    bbox: tuple = (50, 100, 500, 120),
    block_id: str = "",
) -> PDFBlock:
    return PDFBlock(
        page_number=page,
        block_id=block_id or _block_id(page, 0, text),
        block_type=block_type,
        text=text,
        bbox=bbox,
    )


# ── Block ID ─────────────────────────────────────────────────────────────────

def test_block_id_deterministic():
    id1 = _block_id(1, 0, "hello world")
    id2 = _block_id(1, 0, "hello world")
    assert id1 == id2
    assert len(id1) == 16


def test_block_id_differs_for_different_input():
    id1 = _block_id(1, 0, "text A")
    id2 = _block_id(1, 1, "text B")
    assert id1 != id2


# ── Classification ───────────────────────────────────────────────────────────

def test_classify_title():
    block = _make_block("TÍTULO I - DAS DISPOSIÇÕES GERAIS", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "title"


def test_classify_section_header():
    block = _make_block("CAPÍTULO II - DA ADMINISTRAÇÃO", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "section_header"


def test_classify_secao_header():
    block = _make_block("SEÇÃO III - DOS DIREITOS", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "section_header"


def test_classify_annex():
    block = _make_block("ANEXO I", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "annex"


def test_classify_noise_short_text():
    block = _make_block("ab", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "noise"


def test_classify_noise_page_number():
    block = _make_block("42", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "noise"


def test_classify_noise_separator():
    block = _make_block("-------------------", bbox=(50, 200, 500, 220))
    assert _classify_block_type(block) == "noise"


def test_classify_footer_by_position():
    block = _make_block("Página 1 de 10", bbox=(50, 780, 500, 800))
    assert _classify_block_type(block, page_height=842.0) == "footer"


def test_classify_header_by_position():
    block = _make_block("Nº/Rev.: 001 Este documento faz parte do sistema", bbox=(50, 20, 500, 40))
    assert _classify_block_type(block, page_height=842.0) == "header"


def test_classify_table():
    block = _make_block("| Col1 | Col2 |\n| --- | --- |\n| val1 | val2 |", bbox=(50, 200, 500, 300))
    assert _classify_block_type(block) == "table"


def test_classify_body_default():
    block = _make_block(
        "Art. 1 - A empresa tem sede na cidade de Porto Alegre.",
        bbox=(50, 200, 500, 220),
    )
    assert _classify_block_type(block) == "body"


def test_classify_footer_no_bbox():
    block = _make_block("Página 3 de 15", bbox=(0, 0, 0, 0))
    assert _classify_block_type(block) == "footer"


def test_classify_header_no_bbox():
    block = _make_block("Este documento faz parte do sistema de gestão", bbox=(0, 0, 0, 0))
    assert _classify_block_type(block) == "header"


def test_classify_blocks_mutates_in_place():
    blocks = [
        _make_block("CAPÍTULO I - DISPOSIÇÕES", bbox=(50, 200, 500, 220)),
        _make_block("Art. 1 - Texto do artigo.", bbox=(50, 250, 500, 270)),
        _make_block("42", bbox=(50, 780, 500, 800)),
    ]
    result = classify_blocks(blocks)
    assert result is blocks  # mutates in place
    assert blocks[0].block_type == "section_header"
    assert blocks[1].block_type == "body"
    assert blocks[2].block_type == "noise"


# ── Repeated header/footer removal ──────────────────────────────────────────

def test_remove_repeated_headers_marks_repeated_text():
    blocks = []
    repeated_text = "Documento Confidencial - Versão 1.0"
    for page in range(1, 6):
        blocks.append(_make_block(repeated_text, page=page, bbox=(50, 20, 500, 40)))
        blocks.append(_make_block(f"Conteúdo da página {page}.", page=page, bbox=(50, 200, 500, 400)))

    result = remove_repeated_headers_footers(blocks)
    repeated_blocks = [b for b in result if b.text == repeated_text]
    assert all(b.block_type == "header" for b in repeated_blocks)


def test_remove_repeated_headers_no_false_positives():
    blocks = [
        _make_block("Texto único na página 1.", page=1),
        _make_block("Outro texto na página 2.", page=2),
        _make_block("Mais texto na página 3.", page=3),
    ]
    result = remove_repeated_headers_footers(blocks)
    assert all(b.block_type == "body" for b in result)


# ── Paragraph recomposition ──────────────────────────────────────────────────

def test_recompose_merges_broken_paragraph():
    blocks = [
        _make_block("Esta é uma frase que termina com uma vírgula,", page=1, bbox=(50, 100, 500, 120)),
        _make_block("e continua na próxima linha.", page=1, bbox=(50, 120, 500, 140)),
    ]
    result = recompose_paragraphs(blocks)
    assert len(result) == 1
    assert "vírgula, e continua" in result[0].text


def test_recompose_does_not_merge_across_pages():
    blocks = [
        _make_block("Fim da frase,", page=1, bbox=(50, 700, 500, 720)),
        _make_block("início da próxima.", page=2, bbox=(50, 100, 500, 120)),
    ]
    result = recompose_paragraphs(blocks)
    assert len(result) == 2


def test_recompose_skips_non_body_blocks():
    blocks = [
        _make_block("CAPÍTULO I", page=1, block_type="section_header"),
        _make_block("Art. 1 - Texto do artigo.", page=1),
    ]
    result = recompose_paragraphs(blocks)
    assert len(result) == 2
    assert result[0].block_type == "section_header"


def test_recompose_does_not_merge_complete_sentences():
    blocks = [
        _make_block("Esta é uma frase completa.", page=1, bbox=(50, 100, 500, 120)),
        _make_block("Esta é outra frase.", page=1, bbox=(50, 140, 500, 160)),
    ]
    result = recompose_paragraphs(blocks)
    assert len(result) == 2


# ── Section hint assignment ──────────────────────────────────────────────────

def test_assign_section_hints():
    blocks = [
        _make_block("CAPÍTULO I - DISPOSIÇÕES", block_type="section_header"),
        _make_block("Art. 1 - Texto."),
        _make_block("Art. 2 - Mais texto."),
        _make_block("CAPÍTULO II - ADMINISTRAÇÃO", block_type="section_header"),
        _make_block("Art. 10 - Admin."),
    ]
    result = assign_section_hints(blocks)
    assert result[1].section_hint == "CAPÍTULO I - DISPOSIÇÕES"
    assert result[2].section_hint == "CAPÍTULO I - DISPOSIÇÕES"
    assert result[4].section_hint == "CAPÍTULO II - ADMINISTRAÇÃO"


# ── Clean text reconstruction ────────────────────────────────────────────────

def test_build_clean_text_excludes_noise():
    blocks = [
        _make_block("CAPÍTULO I", block_type="section_header", page=1),
        _make_block("Art. 1 - Texto.", page=1),
        _make_block("42", block_type="noise", page=1),
        _make_block("Página 1 de 10", block_type="footer", page=1),
    ]
    text = build_clean_text(blocks)
    assert "CAPÍTULO I" in text
    assert "Art. 1" in text
    assert "42" not in text.split("\n")  # 42 not as standalone
    assert "Página 1 de 10" not in text


def test_build_clean_text_adds_page_breaks():
    blocks = [
        _make_block("Texto página 1.", page=1),
        _make_block("Texto página 2.", page=2),
    ]
    text = build_clean_text(blocks)
    assert "<!-- PAGE_BREAK -->" in text


# ── Artifact generation ──────────────────────────────────────────────────────

def test_generate_markdown_structure():
    blocks = [
        _make_block("TÍTULO I", block_type="title", page=1),
        _make_block("CAPÍTULO I - DISPOSIÇÕES", block_type="section_header", page=1),
        _make_block("Art. 1 - Texto do artigo.", page=1),
    ]
    md = generate_markdown(blocks, "doc_teste")
    assert "# doc_teste" in md
    assert "## TÍTULO I" in md
    assert "### CAPÍTULO I" in md
    assert "Art. 1" in md


def test_generate_json_structure():
    blocks = [_make_block("Art. 1 - Texto.", page=1)]
    quality = ParseQualityResult(score=0.85, total_blocks=1)
    result = generate_json(blocks, "doc_teste", quality)
    assert result["doc_id"] == "doc_teste"
    assert result["total_blocks"] == 1
    assert result["quality"]["score"] == 0.85
    assert len(result["blocks"]) == 1
    assert result["blocks"][0]["text"] == "Art. 1 - Texto."


def test_save_artifacts_creates_files(tmp_path):
    blocks = [_make_block("Conteúdo do documento.", page=1)]
    quality = ParseQualityResult(score=0.9, total_blocks=1)

    md_path, json_path = save_artifacts(blocks, "test_doc", quality, base_dir=str(tmp_path))

    assert Path(md_path).exists()
    assert Path(json_path).exists()
    assert Path(md_path).read_text(encoding="utf-8").startswith("# test_doc")
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert data["doc_id"] == "test_doc"


# ── Quality scoring ──────────────────────────────────────────────────────────

def test_quality_score_empty_blocks():
    quality = compute_quality_score([])
    assert quality.score == 0.0
    assert "no_blocks" in quality.flags
    assert quality.passed is False


def test_quality_score_good_document():
    blocks = [
        _make_block("TÍTULO I - DISPOSIÇÕES", block_type="title", page=1),
        _make_block("CAPÍTULO I - DA EMPRESA", block_type="section_header", page=1),
        _make_block("Art. 1 - A empresa tem sede na cidade de Porto Alegre.", page=1),
        _make_block("Art. 2 - O objeto social é a prestação de serviços.", page=1),
        _make_block("CAPÍTULO II - DO CAPITAL", block_type="section_header", page=2),
        _make_block("Art. 3 - O capital social é de R$ 1.000.000.", page=2),
    ]
    quality = compute_quality_score(blocks)
    assert quality.score > 0.7
    assert quality.structure_detected is True
    assert quality.noise_ratio == 0.0


def test_quality_score_noisy_document():
    blocks = [
        _make_block("ab", block_type="noise", page=1),
        _make_block("", block_type="noise", page=1),
        _make_block("---", block_type="noise", page=1),
        _make_block("Art. 1 - Texto.", page=1),
    ]
    quality = compute_quality_score(blocks)
    assert quality.noise_ratio > 0.5
    assert "high_noise" in quality.flags


def test_quality_score_repetitive_document():
    blocks = []
    for page in range(1, 8):
        blocks.append(_make_block("Texto repetido em todas as paginas.", page=page))
    quality = compute_quality_score(blocks)
    assert quality.repetition_ratio > 0.5
    assert "high_repetition" in quality.flags


def test_quality_score_no_structure():
    blocks = [
        _make_block("Parágrafo um.", page=1),
        _make_block("Parágrafo dois.", page=1),
        _make_block("Parágrafo três.", page=1),
    ]
    quality = compute_quality_score(blocks)
    assert quality.structure_detected is False
    assert "no_structure" in quality.flags


# ── Quality gate ─────────────────────────────────────────────────────────────

def test_quality_gate_success():
    quality = ParseQualityResult(score=0.85, total_blocks=10)
    status = quality_gate(quality, min_score=0.4)
    assert status == "success"
    assert quality.passed is True


def test_quality_gate_needs_review_score():
    quality = ParseQualityResult(score=0.50, total_blocks=10)
    status = quality_gate(quality, min_score=0.4)
    assert status == "needs_review"
    assert quality.passed is True


def test_quality_gate_needs_review_flags():
    quality = ParseQualityResult(score=0.80, flags=["high_noise"], total_blocks=10)
    status = quality_gate(quality, min_score=0.4)
    assert status == "needs_review"
    assert quality.passed is True


def test_quality_gate_failed():
    quality = ParseQualityResult(score=0.2, total_blocks=5)
    status = quality_gate(quality, min_score=0.4)
    assert status == "failed_parse"
    assert quality.passed is False


# ── Integration: ingestion with pipeline ─────────────────────────────────────

def test_ingestion_error_on_low_quality(monkeypatch, tmp_path):
    """PDF pipeline should raise IngestionError when quality is too low."""
    from src.ingestion import IngestionError, _ingest_pdf_pipeline

    monkeypatch.setattr("src.ingestion.log_event", lambda *args, **kwargs: None)

    def fake_process_pdf(**kwargs):
        return PipelineResult(
            blocks=[],
            clean_text="",
            quality=ParseQualityResult(score=0.1, flags=["no_blocks_extracted"], passed=False, total_blocks=0),
            status="failed_parse",
        )

    monkeypatch.setattr("src.pdf_pipeline.process_pdf", fake_process_pdf)

    settings = type("S", (), {
        "pdf_pipeline_save_artifacts": False,
        "pdf_pipeline_artifacts_dir": str(tmp_path),
        "pdf_pipeline_min_quality": 0.4,
    })()

    with pytest.raises(IngestionError) as exc_info:
        _ingest_pdf_pipeline(str(tmp_path / "fake.pdf"), "fake", settings)
    assert exc_info.value.status == "failed_parse"


def test_ingestion_pipeline_returns_clean_text(monkeypatch, tmp_path):
    """PDF pipeline should return clean text for chunking."""
    from src.ingestion import _ingest_pdf_pipeline

    monkeypatch.setattr("src.ingestion.log_event", lambda *args, **kwargs: None)

    def fake_process_pdf(**kwargs):
        return PipelineResult(
            blocks=[_make_block("Texto limpo.")],
            clean_text="Texto limpo para chunking.",
            quality=ParseQualityResult(score=0.85, total_blocks=1),
            md_path="/tmp/doc.md",
            json_path="/tmp/doc.json",
            status="success",
        )

    monkeypatch.setattr("src.pdf_pipeline.process_pdf", fake_process_pdf)

    settings = type("S", (), {
        "pdf_pipeline_save_artifacts": True,
        "pdf_pipeline_artifacts_dir": str(tmp_path),
        "pdf_pipeline_min_quality": 0.4,
    })()

    clean, raw, meta, blocks = _ingest_pdf_pipeline(str(tmp_path / "doc.pdf"), "doc", settings)
    assert clean == "Texto limpo para chunking."
    assert meta["parser"] == "pdf_pipeline"
    assert meta["quality_score"] == 0.85
    assert len(blocks) == 1
