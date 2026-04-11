from __future__ import annotations

from fastapi.testclient import TestClient

import app as app_module
from src.medical_catalog.service import get_medical_catalog_service


client = TestClient(app_module.app)


def test_reindex_route_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDICAL_CATALOG_STORAGE_DIR", str(tmp_path))
    get_medical_catalog_service.cache_clear()

    response = client.post(
        "/reindex",
        json={
            "catalog_id": "rol_api",
            "pdf_path": "data/docs/rol_procedimentos_teste.pdf",
            "config_path": "data/catalog_configs/rol_procedimentos_teste.json",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed_count"] == 5


def test_get_procedure_route(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDICAL_CATALOG_STORAGE_DIR", str(tmp_path))
    get_medical_catalog_service.cache_clear()

    client.post(
        "/reindex",
        json={
            "catalog_id": "rol_api",
            "pdf_path": "data/docs/rol_procedimentos_teste.pdf",
            "config_path": "data/catalog_configs/rol_procedimentos_teste.json",
        },
    )

    response = client.get("/procedure/10101039", params={"catalog_id": "rol_api"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["record"]["descricao"] == "Atendimento em pronto socorro"


def test_ask_route_returns_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDICAL_CATALOG_STORAGE_DIR", str(tmp_path))
    get_medical_catalog_service.cache_clear()

    client.post(
        "/reindex",
        json={
            "catalog_id": "rol_api",
            "pdf_path": "data/docs/rol_procedimentos_teste.pdf",
            "config_path": "data/catalog_configs/rol_procedimentos_teste.json",
        },
    )

    response = client.post(
        "/ask",
        json={
            "catalog_id": "rol_api",
            "question": "Quais procedimentos sao de emergencia?",
            "filters": {"emergencia": "sim"},
            "limit": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["records"]
    assert payload["sources"][0]["page_number"] == 1
