from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class CatalogIngestionConfig(BaseModel):
    catalog_id: str = "default"
    expected_columns: list[str]
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    required_columns: list[str] = Field(default_factory=list)
    code_column: str = "codigo"
    description_column: str = "descricao"
    search_columns: list[str] = Field(default_factory=list)
    filterable_columns: list[str] = Field(default_factory=list)
    code_pattern: str = r"^\d{4,}$"
    pages: str = "all"
    start_after: str | None = None
    stop_after: str | None = None

    @model_validator(mode="after")
    def _fill_defaults(self) -> "CatalogIngestionConfig":
        if not self.required_columns:
            self.required_columns = [self.code_column, self.description_column]
        if not self.search_columns:
            self.search_columns = [self.code_column, self.description_column, *self.filterable_columns]
        return self


class SourceReference(BaseModel):
    source_file: str
    page_number: int
    extractor: str
    excerpt: str = ""


class ProcedureRecord(BaseModel):
    codigo: str
    descricao: str
    fields: dict[str, str]
    source: SourceReference
    searchable_text: str


class CatalogIndex(BaseModel):
    catalog_id: str
    pdf_path: str
    config: CatalogIngestionConfig
    records: list[ProcedureRecord]
    extracted_at: str
    extractor_chain: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReindexRequest(BaseModel):
    catalog_id: str = "default"
    pdf_path: str
    config_path: str | None = None
    config: CatalogIngestionConfig | None = None

    @model_validator(mode="after")
    def _ensure_config_source(self) -> "ReindexRequest":
        if self.config is None and not self.config_path:
            raise ValueError("Informe config ou config_path para reindexar o catálogo.")
        return self


class ReindexResponse(BaseModel):
    catalog_id: str
    pdf_path: str
    indexed_count: int
    extractor_chain: list[str]
    warnings: list[str] = Field(default_factory=list)
    storage_path: str


class AskRequest(BaseModel):
    catalog_id: str = "default"
    question: str
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    records: list[ProcedureRecord]
    sources: list[SourceReference]


class ProcedureLookupResponse(BaseModel):
    answer: str
    record: ProcedureRecord | None
    sources: list[SourceReference]


class CatalogSearchResult(BaseModel):
    record: ProcedureRecord
    score: float


def load_catalog_config(path: str | Path) -> CatalogIngestionConfig:
    raw = Path(path).read_text(encoding="utf-8")
    return CatalogIngestionConfig.model_validate_json(raw)
