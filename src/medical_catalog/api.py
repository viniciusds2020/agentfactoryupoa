from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.medical_catalog.schemas import AskRequest, AskResponse, ProcedureLookupResponse, ReindexRequest, ReindexResponse
from src.medical_catalog.service import get_medical_catalog_service

router = APIRouter(tags=["medical-catalog"])


@router.get("/procedure/{codigo}", response_model=ProcedureLookupResponse)
def get_procedure(codigo: str, catalog_id: str = "default"):
    service = get_medical_catalog_service()
    try:
        return service.get_procedure(catalog_id=catalog_id, codigo=codigo)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/ask", response_model=AskResponse)
def ask_catalog(request: AskRequest):
    service = get_medical_catalog_service()
    try:
        return service.ask(
            catalog_id=request.catalog_id,
            question=request.question,
            filters=request.filters,
            limit=request.limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/reindex", response_model=ReindexResponse)
def reindex_catalog(request: ReindexRequest):
    service = get_medical_catalog_service()
    response = service.reindex(request)
    if response.indexed_count == 0:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Nenhum registro foi indexado a partir do PDF informado.",
                "warnings": response.warnings,
            },
        )
    return response
