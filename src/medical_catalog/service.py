from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from src.medical_catalog.extraction import extract_records_from_pdf
from src.medical_catalog.schemas import (
    AskResponse,
    CatalogIndex,
    CatalogSearchResult,
    ProcedureLookupResponse,
    ProcedureRecord,
    ReindexRequest,
    ReindexResponse,
    load_catalog_config,
)
from src.medical_catalog.store import CatalogStore
from src.medical_catalog.text import normalize_text, tokenize


class MedicalCatalogService:
    def __init__(self, store: CatalogStore | None = None) -> None:
        self.store = store or CatalogStore()

    def reindex(self, request: ReindexRequest) -> ReindexResponse:
        config = request.config or load_catalog_config(request.config_path or "")
        config.catalog_id = request.catalog_id
        extraction = extract_records_from_pdf(request.pdf_path, config)
        index = CatalogIndex(
            catalog_id=request.catalog_id,
            pdf_path=str(Path(request.pdf_path)),
            config=config,
            records=extraction.records,
            extracted_at=self.store.now_iso(),
            extractor_chain=extraction.extractor_chain,
            warnings=extraction.warnings,
        )
        storage_path = self.store.save(index)
        return ReindexResponse(
            catalog_id=request.catalog_id,
            pdf_path=request.pdf_path,
            indexed_count=len(index.records),
            extractor_chain=index.extractor_chain,
            warnings=index.warnings,
            storage_path=str(storage_path),
        )

    def get_procedure(self, catalog_id: str, codigo: str) -> ProcedureLookupResponse:
        index = self.store.load(catalog_id)
        target = self._normalize_code(codigo)
        record = next((item for item in index.records if self._normalize_code(item.codigo) == target), None)
        if record is None:
            return ProcedureLookupResponse(
                answer="Não encontrei esse código no catálogo indexado.",
                record=None,
                sources=[],
            )
        return ProcedureLookupResponse(
            answer=f"Procedimento {record.codigo}: {record.descricao}.",
            record=record,
            sources=[record.source],
        )

    def ask(self, catalog_id: str, question: str, filters: dict[str, str] | None = None, limit: int = 5) -> AskResponse:
        index = self.store.load(catalog_id)
        normalized_filters = self._infer_filters(index.config.filterable_columns, question)
        normalized_filters.update({normalize_text(k): normalize_text(v) for k, v in (filters or {}).items()})
        code = self._extract_code(question)
        records: list[ProcedureRecord] = []
        if code:
            exact = [item for item in index.records if self._normalize_code(item.codigo) == self._normalize_code(code)]
            records = self._apply_filters(exact, normalized_filters)
            if not records:
                return AskResponse(
                    answer="Não encontrei informação suficiente no catálogo para responder com segurança.",
                    records=[],
                    sources=[],
                )
        if not records:
            ranked = self.search(index.records, question, normalized_filters, limit=limit)
            records = [item.record for item in ranked]

        if not records:
            return AskResponse(
                answer="Não encontrei informação suficiente no catálogo para responder com segurança.",
                records=[],
                sources=[],
            )

        if len(records) == 1:
            record = records[0]
            fields = ", ".join(f"{key}={value}" for key, value in record.fields.items() if value)
            answer = f"Resultado objetivo: {record.codigo} - {record.descricao}. Campos encontrados: {fields}."
        else:
            preview = "; ".join(f"{item.codigo} - {item.descricao}" for item in records[:limit])
            answer = f"Resultado objetivo: encontrei {len(records)} registros relevantes. Exemplos: {preview}."

        return AskResponse(
            answer=answer,
            records=records[:limit],
            sources=[item.source for item in records[:limit]],
        )

    def search(
        self,
        records: list[ProcedureRecord],
        question: str,
        filters: dict[str, str],
        limit: int = 5,
    ) -> list[CatalogSearchResult]:
        query_tokens = tokenize(question)
        query_text = normalize_text(question)
        ranked: list[CatalogSearchResult] = []

        for record in self._apply_filters(records, filters):
            haystack = normalize_text(record.searchable_text)
            if not haystack:
                continue
            exact_bonus = 3.0 if query_text and query_text in haystack else 0.0
            token_hits = sum(1 for token in query_tokens if token in haystack)
            if token_hits == 0 and exact_bonus == 0:
                continue
            ranked.append(CatalogSearchResult(record=record, score=exact_bonus + float(token_hits)))

        ranked.sort(key=lambda item: (-item.score, item.record.codigo))
        return ranked[:limit]

    def _apply_filters(self, records: list[ProcedureRecord], filters: dict[str, str]) -> list[ProcedureRecord]:
        if not filters:
            return records
        filtered: list[ProcedureRecord] = []
        for record in records:
            if all(normalize_text(record.fields.get(key, "")) == value for key, value in filters.items()):
                filtered.append(record)
        return filtered

    @staticmethod
    def _extract_code(question: str) -> str | None:
        match = re.search(r"\b\d{4,}\b", question or "")
        return match.group(0) if match else None

    @staticmethod
    def _normalize_code(code: str) -> str:
        return re.sub(r"\D+", "", code or "")

    @staticmethod
    def _infer_filters(filterable_columns: list[str], question: str) -> dict[str, str]:
        normalized_question = normalize_text(question)
        inferred: dict[str, str] = {}
        if "emergencia" in normalized_question and "emergencia" in filterable_columns:
            inferred["emergencia"] = "sim"
        if "autoriz" in normalized_question and "precisa_autorizacao" in filterable_columns:
            inferred["precisa_autorizacao"] = "sim"
        return inferred


@lru_cache
def get_medical_catalog_service() -> MedicalCatalogService:
    return MedicalCatalogService()
