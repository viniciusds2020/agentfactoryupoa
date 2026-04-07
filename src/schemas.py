"""Pydantic schemas shared across the application."""
from __future__ import annotations

from pydantic import BaseModel

from src.chat import ChatMessage


# ── Health / Ingest ──────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class IngestResponse(BaseModel):
    collection: str
    doc_id: str
    chunks_indexed: int


# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    collection: str
    question: str
    history: list[ChatMessage] = []
    embedding_model: str | None = None
    domain_profile: str | None = None


class SourceOut(BaseModel):
    chunk_id: str
    doc_id: str
    excerpt: str
    score: float
    page_number: int | None = None
    source_filename: str = ""
    citation_label: str = ""
    source_kind: str = ""
    query_summary: str = ""
    result_preview: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    request_id: str


# ── Conversations ────────────────────────────────────────────────────────────


class ConversationOut(BaseModel):
    id: str
    title: str
    collection: str
    embedding_model: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    role: str
    content: str
    sources: list[SourceOut]


class CreateConversationRequest(BaseModel):
    collection: str
    embedding_model: str = ""
    title: str = ""


class RenameRequest(BaseModel):
    title: str


class ChatWithHistoryRequest(BaseModel):
    conversation_id: str | None = None
    collection: str
    question: str
    history: list[ChatMessage] = []
    embedding_model: str | None = None
    domain_profile: str | None = None


class ChatWithHistoryResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: list[SourceOut]
    request_id: str


# ── Settings / Observability ─────────────────────────────────────────────────


class SettingsOut(BaseModel):
    llm_provider: str
    llm_model: str
    embedding_model: str
    retrieval_top_k: int
    default_domain_profile: str
    available_domain_profiles: list[str]


class ObservabilityOut(BaseModel):
    counters: dict[str, int]
    timers_ms: dict[str, dict[str, float]]


class RetrievalEvaluationOut(BaseModel):
    dataset: str
    queries: int
    top_k: int
    vector: dict[str, float]


# ── Workspaces / Documents ───────────────────────────────────────────────────


class WorkspaceOut(BaseModel):
    id: str
    name: str
    api_key: str
    is_default: bool
    created_at: str


class CreateWorkspaceRequest(BaseModel):
    name: str


class DocumentOut(BaseModel):
    id: str
    workspace_id: str
    collection: str
    doc_id: str
    filename: str
    embedding_model: str
    status: str
    chunks_indexed: int
    error: str
    context_hint: str
    created_at: str
    updated_at: str


class UpdateCollectionContextRequest(BaseModel):
    context_hint: str = ""


class CollectionContextOut(BaseModel):
    collection: str
    context_hint: str
    updated_documents: int


# ── Tabular / Semantic profiles ──────────────────────────────────────────────


class TableProfileOut(BaseModel):
    workspace_id: str
    collection: str
    table_name: str
    base_context: str
    subject_label: str
    table_type: str = "analytic"
    created_at: str
    updated_at: str


class ColumnProfileOut(BaseModel):
    workspace_id: str
    collection: str
    column_name: str
    display_name: str
    physical_type: str
    semantic_type: str
    role: str
    unit: str
    aliases: list[str]
    examples: list[str]
    description: str
    cardinality: int
    allowed_operations: list[str]


class CollectionSemanticProfileOut(BaseModel):
    collection: str
    profile: TableProfileOut | None = None
    columns: list[ColumnProfileOut]
    value_catalog: dict[str, list[dict]] = {}


# ── Deadlines ────────────────────────────────────────────────────────────────


class DeadlineFaixaOut(BaseModel):
    faixa: str
    count: int
    pct: float


class DeadlineAlertOut(BaseModel):
    codigo: str
    titulo: str
    prazo: str


class DeadlineReportOut(BaseModel):
    collection: str
    total_procedimentos: int
    faixas: list[DeadlineFaixaOut]
    alertas: list[DeadlineAlertOut]


# ── Collections / Ingestion / Audit ──────────────────────────────────────────


class CollectionStatsOut(BaseModel):
    collection: str
    documents: int
    chunks_indexed: int
    updated_at: str


class IngestionJobOut(BaseModel):
    id: str
    workspace_id: str
    collection: str
    doc_id: str
    filename: str
    embedding_model: str
    status: str
    chunks_indexed: int
    error: str
    created_at: str
    started_at: str
    finished_at: str
    stage: str
    progress_pct: int


class DocumentArtifactsOut(BaseModel):
    doc_id: str
    markdown_path: str
    json_path: str
    markdown_preview: str
    json_preview: str
    available: bool


class TabularEvaluationOut(BaseModel):
    cases: int
    summary: dict[str, float]
    details: list[dict]
    dataset: str | None = None
    context_hint: str | None = None
    suites: dict[str, dict] | None = None


class AuditEventOut(BaseModel):
    id: int
    workspace_id: str
    actor: str
    event_type: str
    resource_type: str
    resource_id: str
    details: dict
    created_at: str


class CollectionInfo(BaseModel):
    collection: str
    embedding_model: str
