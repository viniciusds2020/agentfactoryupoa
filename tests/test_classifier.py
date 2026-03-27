from src.classifier import classify_document


class _Block:
    def __init__(self, block_type: str):
        self.block_type = block_type


def test_classify_legal_by_patterns():
    text = "CAPITULO I\nArt. 1 - Foo\nArt. 2 - Bar\nSECAO I\nArt. 3 - Baz"
    result = classify_document(text=text, blocks=[])
    assert result.doc_type == "legal"
    assert result.legal_pattern_count >= 3


def test_classify_tabular_by_block_ratio():
    blocks = [_Block("table"), _Block("table"), _Block("body"), _Block("table"), _Block("body")]
    result = classify_document(text="texto simples", blocks=blocks)
    assert result.doc_type == "tabular"
    assert result.table_ratio > 0.4


def test_classify_mixed_by_block_ratio():
    blocks = [_Block("table"), _Block("table"), _Block("body"), _Block("body"), _Block("body")]
    result = classify_document(text="texto simples", blocks=blocks)
    assert result.doc_type == "mixed"
    assert 0.2 <= result.table_ratio <= 0.4


def test_classify_narrative_default():
    result = classify_document(text="Documento narrativo sem estrutura tabular.")
    assert result.doc_type == "narrative"
    assert result.table_ratio == 0.0
