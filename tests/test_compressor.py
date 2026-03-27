"""Tests for contextual compression module."""
from src.compressor import compress_chunks, compress_extractive


def test_compress_extractive_keeps_relevant_sentences():
    query = "prazo para pagamento"
    text = (
        "O empregador deve cumprir os prazos legais. "
        "O pagamento deve ser feito em 10 dias uteis. "
        "As ferias sao de 30 dias. "
        "O prazo para pagamento de rescisao e de 10 dias."
    )
    result = compress_extractive(query, text, max_sentences=2)
    assert "pagamento" in result
    assert "prazo" in result


def test_compress_extractive_short_text_unchanged():
    query = "teste"
    text = "Uma unica frase curta."
    result = compress_extractive(query, text, max_sentences=3)
    assert result == text


def test_compress_extractive_preserves_original_order():
    query = "salario"
    text = (
        "Primeiro paragrafo sobre ferias. "
        "Segundo paragrafo sobre salario minimo. "
        "Terceiro paragrafo sobre jornada. "
        "Quarto paragrafo sobre salario base."
    )
    result = compress_extractive(query, text, max_sentences=2)
    sentences = result.split(" ")
    # "Segundo" should come before "Quarto" in output
    idx_segundo = result.find("Segundo")
    idx_quarto = result.find("Quarto")
    if idx_segundo >= 0 and idx_quarto >= 0:
        assert idx_segundo < idx_quarto


def test_compress_chunks_skips_parent_chunks():
    query = "teste"
    chunks = [
        {"text": "Long text. With many sentences. That could be compressed. Into fewer.", "metadata": {"chunk_type": "parent"}},
        {"text": "Long text. With many sentences. That could be compressed. Into fewer.", "metadata": {"chunk_type": "child"}},
    ]
    result = compress_chunks(query, chunks, method="extractive", max_sentences=2)
    # Parent chunk should be unchanged
    assert result[0]["text"] == chunks[0]["text"]
    # Child chunk may be compressed
    assert len(result[1]["text"]) <= len(chunks[1]["text"])


def test_compress_chunks_empty_list():
    result = compress_chunks("query", [], method="extractive")
    assert result == []


def test_compress_chunks_unknown_method_returns_unchanged():
    chunks = [{"text": "some text", "metadata": {}}]
    result = compress_chunks("query", chunks, method="unknown")
    assert result == chunks
