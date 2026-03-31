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


def test_classify_tabular_by_catalog_signals_and_code_lines(monkeypatch):
    monkeypatch.setattr("src.classifier._table_probe_from_pdf", lambda path: 0)
    text = (
        "Procedimento Descricao Cobertura Prazo Autorizacao\n"
        "20017 Visita Domiciliar Hospitalar 10 dias uteis\n"
        "20101325 Consulta em pronto atendimento Ambulatorial imediato\n"
        "10106103 Outro procedimento Hospitalar imediato\n"
        "10101039 Chamado de especialista Ambulatorial imediato\n"
        "10101055 Teleconsulta Plantao Ambulatorial imediato\n"
        "10101063 Teleconsulta Eletiva Ambulatorial nao\n"
        "10101020 Consulta em domicilio Sem Cobertura\n"
        "10101012 Consulta em consultorio Ambulatorial\n"
    )
    result = classify_document(text=text, blocks=[])
    assert result.doc_type == "tabular"
    assert result.catalog_signal_count >= 3
    assert result.code_line_count >= 8


def test_classify_mixed_by_table_probe_rows(monkeypatch):
    monkeypatch.setattr("src.classifier._table_probe_from_pdf", lambda path: 4)
    result = classify_document(text="relatorio com tabela parcial", blocks=[], pdf_path="fake.pdf")
    assert result.doc_type == "mixed"
    assert result.table_probe_rows == 4
