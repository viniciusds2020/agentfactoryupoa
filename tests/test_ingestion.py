"""Tests for ingestion chunking logic — no external I/O."""
from src.ingestion import _split, _split_sentences


def test_split_single_chunk_for_short_text():
    text = "Texto curto."
    chunks = _split(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_no_empty_chunks():
    text = "  \n  ".join(["palavra"] * 300)
    chunks = _split(text)
    assert all(len(c) > 0 for c in chunks)


def test_split_respects_chunk_size():
    text = ". ".join(f"Frase número {i} com conteúdo relevante" for i in range(100))
    chunks = _split(text, chunk_size=200)
    for chunk in chunks:
        assert len(chunk) <= 200 + 50  # small tolerance for sentence packing


def test_split_sentences_basic():
    text = "Primeira frase. Segunda frase! Terceira frase?"
    sentences = _split_sentences(text)
    assert len(sentences) == 3
    assert "Primeira frase." in sentences[0]
    assert "Segunda frase!" in sentences[1]


def test_split_sentences_preserves_abbreviations():
    text = "Art. 477 da CLT estabelece prazos. O Sr. João concordou."
    sentences = _split_sentences(text)
    assert any("Art. 477" in s for s in sentences)
    assert any("Sr. João" in s for s in sentences)


def test_split_sentence_aware_respects_boundaries():
    """Chunks should not split in the middle of a sentence."""
    text = "Primeira frase curta. Segunda frase curta. Terceira frase curta. Quarta frase curta."
    chunks = _split(text, chunk_size=80, chunk_overlap=20)
    for chunk in chunks:
        # Each chunk should end with punctuation or be the tail of a sentence
        assert chunk[-1] in ".!?;" or chunk == chunks[-1]


def test_split_fallback_for_long_sentence():
    """A single sentence longer than chunk_size should be char-split."""
    long_sentence = "A" * 1000
    chunks = _split(long_sentence, chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 5
    for chunk in chunks:
        assert len(chunk) <= 200


def test_split_sentence_aware_overlap():
    """Overlap should carry trailing sentences from previous chunk."""
    text = "Frase um. Frase dois. Frase três. Frase quatro. Frase cinco. Frase seis."
    chunks = _split(text, chunk_size=40, chunk_overlap=20)
    assert len(chunks) >= 2
    # Check that some content from end of chunk N appears at start of chunk N+1
    if len(chunks) > 1:
        last_words_chunk0 = chunks[0].split()[-2:]
        first_words_chunk1 = chunks[1].split()[:3]
        overlap_found = any(w in first_words_chunk1 for w in last_words_chunk0)
        assert overlap_found, f"No overlap found between '{chunks[0]}' and '{chunks[1]}'"
