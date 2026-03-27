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
