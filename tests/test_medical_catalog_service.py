from __future__ import annotations

from pathlib import Path

from src.medical_catalog.schemas import ReindexRequest
from src.medical_catalog.service import MedicalCatalogService
from src.medical_catalog.store import CatalogStore


FIXTURE_PDF = Path("data/docs/rol_procedimentos_teste.pdf")
FIXTURE_CONFIG = Path("data/catalog_configs/rol_procedimentos_teste.json")


def _service(tmp_path):
    return MedicalCatalogService(store=CatalogStore(tmp_path))


def test_reindex_sample_pdf(tmp_path):
    service = _service(tmp_path)

    result = service.reindex(
        ReindexRequest(
            catalog_id="rol_teste",
            pdf_path=str(FIXTURE_PDF),
            config_path=str(FIXTURE_CONFIG),
        )
    )

    assert result.indexed_count == 5
    assert "pymupdf_text" in result.extractor_chain


def test_lookup_by_code(tmp_path):
    service = _service(tmp_path)
    service.reindex(
        ReindexRequest(
            catalog_id="rol_teste",
            pdf_path=str(FIXTURE_PDF),
            config_path=str(FIXTURE_CONFIG),
        )
    )

    response = service.get_procedure("rol_teste", "20202020")

    assert response.record is not None
    assert response.record.codigo == "20202020"
    assert "Cirurgia eletiva de joelho" in response.answer
    assert response.sources[0].page_number == 1


def test_ask_uses_catalog_content_only(tmp_path):
    service = _service(tmp_path)
    service.reindex(
        ReindexRequest(
            catalog_id="rol_teste",
            pdf_path=str(FIXTURE_PDF),
            config_path=str(FIXTURE_CONFIG),
        )
    )

    response = service.ask("rol_teste", "Quais procedimentos sao de emergencia?", limit=3)

    assert response.records
    assert all(record.fields["emergencia"] == "sim" for record in response.records)
    assert response.sources[0].source_file == FIXTURE_PDF.name


def test_ask_returns_safe_answer_when_missing(tmp_path):
    service = _service(tmp_path)
    service.reindex(
        ReindexRequest(
            catalog_id="rol_teste",
            pdf_path=str(FIXTURE_PDF),
            config_path=str(FIXTURE_CONFIG),
        )
    )

    response = service.ask("rol_teste", "Qual o prazo regulatorio do procedimento 99999999?")

    assert response.records == []
    assert "segurança" in response.answer
