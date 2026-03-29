from src.tabular import (
    TableRecord,
    build_texto_canonico,
    chunk_tabular_records,
    extract_tables_from_text,
)


def test_build_texto_canonico_orders_columns():
    fields = {"codigo": "10101039", "descricao": "Procedimento X", "autorizacao": "sim"}
    text = build_texto_canonico(fields, ["codigo", "descricao", "autorizacao"])
    assert "codigo: 10101039" in text
    assert "descricao: Procedimento X" in text


def test_extract_tables_from_text_markdown():
    md = """
| codigo | descricao | emergencia |
| --- | --- | --- |
| 10101039 | Exame A | sim |
| 10101040 | Exame B | nao |
"""
    result = extract_tables_from_text(md)
    assert len(result.records) == 2
    assert result.records[0].fields["codigo"] == "10101039"


def test_extract_tables_from_text_ignores_rows_without_code():
    md = """
| codigo | descricao |
| --- | --- |
| sem_codigo | linha invalida |
| 12345 | linha valida |
"""
    result = extract_tables_from_text(md)
    assert len(result.records) == 1
    assert result.records[0].fields["codigo"] == "12345"


def test_extract_tables_from_text_accepts_csv_like_rows_with_short_ids():
    md = """
| id_cliente | nome | idade | renda_mensal | estado |
| --- | --- | --- | --- | --- |
| 1 | Ursula Souza | 19 | 26341.94 | MG |
| 2 | Ana Souza | 31 | 9294.14 | SC |
"""
    result = extract_tables_from_text(md)
    assert len(result.records) == 2
    assert result.records[0].fields["id_cliente"] == "1"


def test_chunk_tabular_records_groups_short_rows():
    records = [
        TableRecord(i, 1, {"codigo": str(10000 + i)}, f"codigo: {10000 + i}", f"raw {i}")
        for i in range(6)
    ]
    chunks = chunk_tabular_records(records, group_size=3, max_chunk_chars=200)
    assert len(chunks) == 2
    assert chunks[0][1]["row_count"] == 3
    assert chunks[1][1]["row_count"] == 3


def test_chunk_tabular_records_keeps_long_row_alone():
    records = [
        TableRecord(0, 1, {"codigo": "12345"}, "x" * 700, "raw"),
    ]
    chunks = chunk_tabular_records(records, group_size=5, max_chunk_chars=512)
    assert len(chunks) == 1
    assert chunks[0][1]["row_count"] == 1
    assert len(chunks[0][0]) == 512
