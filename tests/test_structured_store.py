import pytest

duckdb = pytest.importorskip("duckdb")

from src import structured_store


def _settings(path: str):
    return type("S", (), {"structured_store_path": path})()


def test_ensure_table_and_upsert_records(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "10101039 | Exame A",
            "texto_canonico": "codigo: 10101039; descricao: Exame A",
            "fields": {"codigo": "10101039", "descricao": "Exame A"},
        }
    ]
    n = structured_store.upsert_records("rol_procedimentos", "doc-1", records, ["codigo", "descricao"])
    assert n == 1
    assert structured_store.has_structured_data("rol_procedimentos") is True


def test_query_structured_by_filter(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "10101039 | Exame A",
            "texto_canonico": "codigo: 10101039; descricao: Exame A",
            "fields": {"codigo": "10101039", "descricao": "Exame A", "emergencia": "sim"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "10101040 | Exame B",
            "texto_canonico": "codigo: 10101040; descricao: Exame B",
            "fields": {"codigo": "10101040", "descricao": "Exame B", "emergencia": "nao"},
        },
    ]
    structured_store.upsert_records("rol", "doc-1", records, ["codigo", "descricao", "emergencia"])
    rows = structured_store.query_structured("rol", {"codigo": "10101039"})
    assert len(rows) == 1
    assert rows[0]["codigo"] == "10101039"


def test_query_structured_resolves_catalog_alias_to_physical_column(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "20017 | Visita Domiciliar",
            "texto_canonico": "procedimento: 20017; descricao_unimed_poa: Visita Domiciliar",
            "fields": {
                "procedimento": "20017",
                "descricao_unimed_poa": "Visita Domiciliar",
                "cobertura_unimed_poa": "Hospitalar",
            },
        }
    ]
    structured_store.upsert_records(
        "rol_catalogo",
        "doc-1",
        records,
        ["procedimento", "descricao_unimed_poa", "cobertura_unimed_poa"],
    )

    rows = structured_store.query_structured("rol_catalogo", {"codigo": "20017"})

    assert len(rows) == 1
    assert rows[0]["procedimento"] == "20017"


def test_delete_by_doc_id(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "12345 | X",
            "texto_canonico": "codigo: 12345; descricao: X",
            "fields": {"codigo": "12345", "descricao": "X"},
        }
    ]
    structured_store.upsert_records("rol", "doc-2", records, ["codigo", "descricao"])
    deleted = structured_store.delete_by_doc_id("rol", "doc-2")
    assert deleted >= 1


def test_plan_query_distinct_dimension(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | Ana | RJ",
            "texto_canonico": "id_cliente: 1; nome: Ana; estado: RJ",
            "fields": {"id_cliente": "1", "nome": "Ana", "estado": "RJ"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "2 | Bia | SP",
            "texto_canonico": "id_cliente: 2; nome: Bia; estado: SP",
            "fields": {"id_cliente": "2", "nome": "Bia", "estado": "SP"},
        },
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "nome", "estado"])

    plan = structured_store.plan_query("cadastro", "Quais sao os estados que estao na base?")

    assert plan is not None
    assert plan["operation"] == "distinct"
    assert plan["dimension_column"] == "estado"


def test_execute_plan_distinct_dimension(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | Ana | RJ",
            "texto_canonico": "id_cliente: 1; nome: Ana; estado: RJ",
            "fields": {"id_cliente": "1", "nome": "Ana", "estado": "RJ"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "2 | Bia | SP",
            "texto_canonico": "id_cliente: 2; nome: Bia; estado: SP",
            "fields": {"id_cliente": "2", "nome": "Bia", "estado": "SP"},
        },
        {
            "row_index": 2,
            "page_number": 1,
            "raw_row": "3 | Caio | RJ",
            "texto_canonico": "id_cliente: 3; nome: Caio; estado: RJ",
            "fields": {"id_cliente": "3", "nome": "Caio", "estado": "RJ"},
        },
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "nome", "estado"])

    result = structured_store.execute_plan(
        "cadastro",
        {"operation": "distinct", "dimension_column": "estado", "filters": [], "limit": 100},
    )

    assert result is not None
    assert result["operation"] == "distinct"
    assert result["count"] == 2
    assert result["values"] == ["RJ", "SP"]


def test_plan_query_schema_listing(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | Ana | RJ",
            "texto_canonico": "id_cliente: 1; nome: Ana; estado: RJ",
            "fields": {"id_cliente": "1", "nome": "Ana", "estado": "RJ"},
        }
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "nome", "estado"])

    plan = structured_store.plan_query("cadastro", "Quais sao as colunas da tabela?")

    assert plan is not None
    assert plan["operation"] == "schema"
    assert plan["intent"] == "tabular_schema"


def test_execute_plan_schema_listing(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | Ana | RJ",
            "texto_canonico": "id_cliente: 1; nome: Ana; estado: RJ",
            "fields": {"id_cliente": "1", "nome": "Ana", "estado": "RJ"},
        }
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "nome", "estado"])

    result = structured_store.execute_plan("cadastro", {"operation": "schema"})

    assert result is not None
    assert result["operation"] == "schema"
    assert "id_cliente" in result["columns"]
    assert "estado" in result["columns"]
    assert result["profiles"][0]["allowed_operations"]


def test_get_column_profiles_exposes_semantic_type_and_unit(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | Ana | 34 | 1200.0 | RJ",
            "texto_canonico": "id_cliente: 1; nome: Ana; idade: 34; renda_mensal: 1200.0; estado: RJ",
            "fields": {"id_cliente": "1", "nome": "Ana", "idade": "34", "renda_mensal": "1200.0", "estado": "RJ"},
        }
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "nome", "idade", "renda_mensal", "estado"])

    profiles = structured_store.get_column_profiles("cadastro")
    by_name = {item["name"]: item for item in profiles}

    assert by_name["idade"]["semantic_type"] == "measure_age_years"
    assert by_name["idade"]["unit"] == "anos"
    assert by_name["renda_mensal"]["semantic_type"] == "measure_currency"
    assert by_name["estado"]["semantic_type"] == "dimension_geo_state"
    assert "avg" in by_name["idade"]["allowed_operations"]


def test_plan_query_describe_column(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | 750",
            "texto_canonico": "id_cliente: 1; score_credito: 750",
            "fields": {"id_cliente": "1", "score_credito": "750"},
        }
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "score_credito"])

    plan = structured_store.plan_query("cadastro", "O que significa a coluna score_credito?")

    assert plan is not None
    assert plan["operation"] == "describe_column"
    assert plan["target_column"] == "score_credito"


def test_execute_plan_aggregate_exposes_sql_generated(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | 1000 | RJ",
            "texto_canonico": "id_cliente: 1; renda_mensal: 1000; estado: RJ",
            "fields": {"id_cliente": "1", "renda_mensal": "1000", "estado": "RJ"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "2 | 2000 | RJ",
            "texto_canonico": "id_cliente: 2; renda_mensal: 2000; estado: RJ",
            "fields": {"id_cliente": "2", "renda_mensal": "2000", "estado": "RJ"},
        },
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "renda_mensal", "estado"])

    plan = structured_store.plan_query("cadastro", "Qual e a renda total no estado do Rio de Janeiro?")
    result = structured_store.execute_plan("cadastro", plan)

    assert result is not None
    assert "SELECT SUM(" in result["sql_generated"]


def test_plan_query_adds_comparative_age_filter_and_avoids_city_duplication(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | 35 | 1500.50 | Sao Paulo | SP",
            "texto_canonico": "id_cliente: 1; idade: 35; renda_mensal: 1500.50; cidade: Sao Paulo; estado: SP",
            "fields": {"id_cliente": "1", "idade": "35", "renda_mensal": "1500.50", "cidade": "Sao Paulo", "estado": "SP"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "2 | 28 | 2500.75 | Campinas | SP",
            "texto_canonico": "id_cliente: 2; idade: 28; renda_mensal: 2500.75; cidade: Campinas; estado: SP",
            "fields": {"id_cliente": "2", "idade": "28", "renda_mensal": "2500.75", "cidade": "Campinas", "estado": "SP"},
        },
    ]
    structured_store.upsert_records("cadastro", "doc-1", records, ["id_cliente", "idade", "renda_mensal", "cidade", "estado"])

    plan = structured_store.plan_query("cadastro", "Qual e a media de renda em Sao Paulo com pessoas acima de 30 anos?")

    assert plan is not None
    assert plan["aggregation"] == "avg"
    assert any(f["column"] == "idade" and f["operator"] == ">" and str(f["value"]) == "30" for f in plan["filters"])
    assert any(f["column"] == "estado" and f["value"] == "SP" for f in plan["filters"])
    assert not any(f["column"] == "cidade" and f["value"] == "Sao Paulo" for f in plan["filters"])


def test_plan_query_catalog_lookup_by_id(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "20017 | Visita Domiciliar",
            "texto_canonico": "procedimento: 20017; descricao_unimed_poa: Visita Domiciliar; cobertura_unimed_poa: Hospitalar",
            "fields": {
                "procedimento": "20017",
                "descricao_unimed_poa": "Visita Domiciliar",
                "cobertura_unimed_poa": "Hospitalar",
                "orientacao_autorizacao_call_center": "Necessita autorizacao da Unidade",
            },
        }
    ]
    structured_store.upsert_records(
        "rol",
        "doc-1",
        records,
        ["procedimento", "descricao_unimed_poa", "cobertura_unimed_poa", "orientacao_autorizacao_call_center"],
    )

    plan = structured_store.plan_query("rol", "Qual e o procedimento com esse id 20017?", context_hint="Catalogo de procedimentos e cobertura")

    assert plan is not None
    assert plan["operation"] == "catalog_lookup"
    assert plan["intent"] == "catalog_lookup_by_id"
    assert plan["table_type"] == "catalog"
    assert plan["filters"][0]["column"] == "procedimento"
    assert plan["filters"][0]["value"] == "20017"


def test_execute_plan_catalog_field_lookup(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "20017 | Visita Domiciliar",
            "texto_canonico": "procedimento: 20017; descricao_unimed_poa: Visita Domiciliar; cobertura_unimed_poa: Hospitalar",
            "fields": {
                "procedimento": "20017",
                "descricao_unimed_poa": "Visita Domiciliar",
                "cobertura_unimed_poa": "Hospitalar",
                "orientacao_autorizacao_call_center": "Necessita autorizacao da Unidade",
            },
        }
    ]
    structured_store.upsert_records(
        "rol",
        "doc-1",
        records,
        ["procedimento", "descricao_unimed_poa", "cobertura_unimed_poa", "orientacao_autorizacao_call_center"],
    )

    plan = structured_store.plan_query("rol", "Qual a cobertura do 20017?", context_hint="Catalogo de procedimentos")
    result = structured_store.execute_plan("rol", plan)

    assert result is not None
    assert result["operation"] == "catalog_field_lookup"
    assert result["record"]["procedimento"] == "20017"
    assert result["target_column"] == "cobertura_unimed_poa"


def test_plan_query_groupby_month_for_time_series(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | 2025-01-10 | 1000",
            "texto_canonico": "id_venda: 1; data_venda: 2025-01-10; valor_total: 1000",
            "fields": {"id_venda": "1", "data_venda": "2025-01-10", "valor_total": "1000"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "2 | 2025-02-10 | 1500",
            "texto_canonico": "id_venda: 2; data_venda: 2025-02-10; valor_total: 1500",
            "fields": {"id_venda": "2", "data_venda": "2025-02-10", "valor_total": "1500"},
        },
    ]
    structured_store.upsert_records("vendas", "doc-1", records, ["id_venda", "data_venda", "valor_total"])

    plan = structured_store.plan_query("vendas", "Qual a receita por mes?", context_hint="Base temporal de vendas")

    assert plan is not None
    assert plan["intent"] == "tabular_groupby"
    assert plan["time_grain"] == "month"
    assert plan["metric_column"] == "valor_total"


def test_execute_plan_groupby_month_generates_bucketed_sql(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "1 | 2025-01-10 | 1000",
            "texto_canonico": "id_venda: 1; data_venda: 2025-01-10; valor_total: 1000",
            "fields": {"id_venda": "1", "data_venda": "2025-01-10", "valor_total": "1000"},
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "2 | 2025-02-10 | 1500",
            "texto_canonico": "id_venda: 2; data_venda: 2025-02-10; valor_total: 1500",
            "fields": {"id_venda": "2", "data_venda": "2025-02-10", "valor_total": "1500"},
        },
    ]
    structured_store.upsert_records("vendas", "doc-1", records, ["id_venda", "data_venda", "valor_total"])

    plan = structured_store.plan_query("vendas", "Qual a receita por mes?", context_hint="Base temporal de vendas")
    result = structured_store.execute_plan("vendas", plan)

    assert result is not None
    assert result["operation"] == "groupby"
    assert "strftime(" in result["sql_generated"]


def test_upsert_records_populates_deadline_derivatives(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "20017 | Visita Domiciliar",
            "texto_canonico": "procedimento: 20017; prazo_autorizacao: ate 10 dias uteis; orientacao_autorizacao: Necessita autorizacao da Unidade",
            "fields": {
                "procedimento": "20017",
                "descricao_unimed_poa": "Visita Domiciliar",
                "prazo_autorizacao": "Atendimento em regime de internacao eletiva - ate 10 (dez) dias uteis",
                "orientacao_autorizacao": "Necessita autorizacao da Unidade",
            },
        }
    ]
    structured_store.upsert_records(
        "rol_deadline",
        "doc-1",
        records,
        ["procedimento", "descricao_unimed_poa", "prazo_autorizacao", "orientacao_autorizacao"],
    )

    rows = structured_store.query_structured("rol_deadline", {"codigo": "20017"})
    assert rows
    assert rows[0]["prazo_dias"] == 10
    assert rows[0]["prazo_faixa"] == "Ate 10 dias"
    assert rows[0]["prazo_urgencia"] == "medio"
    assert rows[0]["requer_autorizacao"] == "true"


def test_plan_query_catalog_filter_for_emergencia(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "10101039 | Consulta em pronto socorro",
            "texto_canonico": "procedimento: 10101039; descricao_unimed_poa: Consulta em pronto socorro; emergencia: SIM",
            "fields": {"procedimento": "10101039", "descricao_unimed_poa": "Consulta em pronto socorro", "emergencia": "SIM"},
        }
    ]
    structured_store.upsert_records("rol_emergencia", "doc-1", records, ["procedimento", "descricao_unimed_poa", "emergencia"])

    plan = structured_store.plan_query("rol_emergencia", "Quais procedimentos sao de emergencia?", context_hint="Catalogo de procedimentos")

    assert plan is not None
    assert plan["intent"] == "catalog_filter"
    assert any(f["column"] == "emergencia" and f["value"] == "SIM" for f in plan["filters"])


def test_execute_plan_catalog_deadline_report(tmp_path, monkeypatch):
    db_path = str(tmp_path / "structured.duckdb")
    monkeypatch.setattr("src.structured_store.get_settings", lambda: _settings(db_path))
    structured_store._CONN = None
    structured_store._BACKEND = ""

    records = [
        {
            "row_index": 0,
            "page_number": 1,
            "raw_row": "10101039 | Consulta em pronto socorro",
            "texto_canonico": "procedimento: 10101039; descricao_unimed_poa: Consulta em pronto socorro; prazo_autorizacao: Urgencia/Emergencia - imediato",
            "fields": {
                "procedimento": "10101039",
                "descricao_unimed_poa": "Consulta em pronto socorro",
                "prazo_autorizacao": "Urgencia/Emergencia - imediato",
            },
        },
        {
            "row_index": 1,
            "page_number": 1,
            "raw_row": "20017 | Visita Domiciliar",
            "texto_canonico": "procedimento: 20017; descricao_unimed_poa: Visita Domiciliar; prazo_autorizacao: ate 10 dias uteis",
            "fields": {
                "procedimento": "20017",
                "descricao_unimed_poa": "Visita Domiciliar",
                "prazo_autorizacao": "Atendimento em regime de internacao eletiva - ate 10 (dez) dias uteis",
            },
        },
    ]
    structured_store.upsert_records("rol_report", "doc-1", records, ["procedimento", "descricao_unimed_poa", "prazo_autorizacao"])

    plan = structured_store.plan_query("rol_report", "Relatorio de prazos", context_hint="Catalogo de procedimentos")
    result = structured_store.execute_plan("rol_report", plan)

    assert result is not None
    assert result["operation"] == "catalog_deadline_report"
    assert result["report"]["total_procedimentos"] == 2
    assert any(item["faixa"] == "Imediato" for item in result["report"]["faixas"])
    assert any(item["codigo"] == "10101039" for item in result["report"]["alertas"])
