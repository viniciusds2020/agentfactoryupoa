from src.tabular import (
    TableRecord,
    build_texto_canonico,
    chunk_tabular_records,
    extract_tables_from_text,
)
from src.table_planner import build_query_plan
from src.table_renderer import render_table_answer
from src.table_semantics import infer_profiles_from_records, infer_table_type


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


def test_extract_tables_from_text_accepts_identifierish_catalog_rows():
    md = """
| procedimento | descricao | cobertura |
| --- | --- | --- |
| A12 | Exame especial | Hospitalar |
| B34 | Outro exame | Ambulatorial |
"""
    result = extract_tables_from_text(md)
    assert len(result.records) == 2
    assert result.records[0].fields["procedimento"] == "A12"


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


def test_chunk_tabular_records_can_force_single_row_for_catalog():
    records = [
        TableRecord(0, 1, {"procedimento": "10001", "descricao": "Item A"}, "procedimento: 10001; descricao: Item A", "raw 1"),
        TableRecord(1, 1, {"procedimento": "10002", "descricao": "Item B"}, "procedimento: 10002; descricao: Item B", "raw 2"),
    ]
    chunks = chunk_tabular_records(
        records,
        group_size=5,
        max_chunk_chars=512,
        force_single_row=True,
        header_columns=["procedimento", "descricao"],
    )
    assert len(chunks) == 2
    assert chunks[0][1]["row_count"] == 1
    assert chunks[1][1]["row_count"] == 1
    assert "colunas: procedimento | descricao" in chunks[0][0]
    assert "procedimento: 10001" in chunks[0][0]


def test_infer_table_type_from_catalog_like_records():
    records = [
        TableRecord(0, 1, {"procedimento": "10001", "descricao": "Item A", "prazo": "2 dias", "autorizacao": "Sim"}, "", ""),
        TableRecord(1, 1, {"procedimento": "10002", "descricao": "Item B", "prazo": "5 dias", "autorizacao": "Nao"}, "", ""),
    ]
    profiles = infer_profiles_from_records(["procedimento", "descricao", "prazo", "autorizacao"], records)
    assert infer_table_type(profiles) == "catalog"


def test_infer_semantic_emergencia():
    profiles = infer_profiles_from_records(
        ["procedimento", "descricao", "emergencia"],
        [TableRecord(0, 1, {"procedimento": "1", "descricao": "X", "emergencia": "SIM"}, "", "")],
    )
    by_name = {item["name"]: item for item in profiles}
    assert by_name["emergencia"]["semantic_type"] == "flag_boolean"


def test_infer_semantic_segmentacao():
    profiles = infer_profiles_from_records(
        ["procedimento", "segmentacao_ans"],
        [TableRecord(0, 1, {"procedimento": "1", "segmentacao_ans": "Hospitalar"}, "", "")],
    )
    by_name = {item["name"]: item for item in profiles}
    assert by_name["segmentacao_ans"]["semantic_type"] == "dimension_category"


def test_catalog_filter_plan_emergencia():
    profiles = [
        {"name": "procedimento", "role": "identifier", "semantic_type": "identifier", "unit": "text", "aliases": ["procedimento", "codigo"]},
        {"name": "descricao_unimed_poa", "role": "text", "semantic_type": "catalog_title", "unit": "text", "aliases": ["descricao"]},
        {"name": "emergencia", "role": "dimension", "semantic_type": "flag_boolean", "unit": "boolean", "aliases": ["emergencia", "urgencia"]},
    ]
    plan = build_query_plan("Quais procedimentos sao de emergencia?", profiles, [], context_hint="Catalogo de procedimentos")
    assert plan is not None
    assert plan.intent == "catalog_filter"
    assert any(f.column == "emergencia" and f.value == "SIM" for f in plan.filters)


def test_catalog_filter_plan_authorization():
    profiles = [
        {"name": "procedimento", "role": "identifier", "semantic_type": "identifier", "unit": "text", "aliases": ["procedimento", "codigo"]},
        {"name": "orientacao_autorizacao", "role": "text", "semantic_type": "authorization_rule", "unit": "text", "aliases": ["autorizacao"]},
    ]
    plan = build_query_plan("Quais procedimentos precisam de autorizacao?", profiles, [], context_hint="Catalogo de procedimentos")
    assert plan is not None
    assert plan.intent == "catalog_filter"


def test_catalog_filter_plan_coverage():
    profiles = [
        {"name": "procedimento", "role": "identifier", "semantic_type": "identifier", "unit": "text", "aliases": ["procedimento", "codigo"]},
        {"name": "cobertura_unimed_poa", "role": "text", "semantic_type": "coverage_rule", "unit": "text", "aliases": ["cobertura"]},
    ]
    plan = build_query_plan("Quais procedimentos tem cobertura?", profiles, [], context_hint="Catalogo de procedimentos")
    assert plan is not None
    assert plan.intent == "catalog_filter"


def test_catalog_deadline_report_plan():
    profiles = [
        {"name": "procedimento", "role": "identifier", "semantic_type": "identifier", "unit": "text", "aliases": ["procedimento", "codigo"]},
        {"name": "prazo_autorizacao", "role": "text", "semantic_type": "deadline_rule", "unit": "text", "aliases": ["prazo", "prazo autorizacao"]},
    ]
    plan = build_query_plan("Relatorio de prazos", profiles, [], context_hint="Catalogo de procedimentos")
    assert plan is not None
    assert plan.intent == "catalog_deadline_report"


def test_render_catalog_filter_results():
    answer, _ = render_table_answer(
        question="Quais procedimentos sao de emergencia?",
        plan={"intent": "catalog_filter"},
        result={
            "rows": [
                {"procedimento": "10101039", "descricao_unimed_poa": "Consulta em pronto socorro", "cobertura_unimed_poa": "Ambulatorial"},
                {"procedimento": "10101047", "descricao_unimed_poa": "Chamado de especialista", "cobertura_unimed_poa": "Ambulatorial"},
            ],
            "profiles": [
                {"name": "procedimento", "semantic_type": "identifier"},
                {"name": "descricao_unimed_poa", "semantic_type": "catalog_title"},
                {"name": "cobertura_unimed_poa", "semantic_type": "coverage_rule"},
            ],
        },
        context_hint="Catalogo",
        business_scope="",
    )
    assert "Encontrei 2 procedimentos" in answer
    assert "10101039 - Consulta em pronto socorro" in answer


def test_render_deadline_report():
    answer, _ = render_table_answer(
        question="Relatorio de prazos",
        plan={"intent": "catalog_deadline_report"},
        result={"report": {"faixas": [{"faixa": "Imediato", "count": 3, "pct": 12.0}], "total_procedimentos": 25}},
        context_hint="Catalogo",
        business_scope="",
    )
    assert "Relatorio de prazos do catalogo" in answer
    assert "- Imediato: 3 procedimentos (12%)" in answer
