"""Agent Factory Lite - FastAPI entry point."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import pathlib
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.chat import ChatMessage, ChatResult, StreamContext, answer, prepare_stream
from src.config import get_settings
from src.controlplane import (
    create_ingestion_job,
    create_workspace,
    delete_document_record,
    ensure_default_workspace,
    get_document,
    get_ingestion_job,
    list_audit_events,
    list_collection_stats,
    list_collections_with_models,
    list_documents as list_document_records,
    list_ingestion_jobs,
    list_workspaces,
    record_audit,
    resolve_workspace,
    update_ingestion_job,
    upsert_document,
)
from src.prompts import list_domain_profiles
from src.evaluation import evaluate_retrieval_snapshot
from src.guardrails import chat_limiter, ingest_limiter, sanitize_question, validate_collection
from src.observability import metrics
from src.utils import get_logger, log_event, new_request_id, request_id_var

logger = get_logger(__name__)
settings = get_settings()

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".xlsx", ".xls", ".csv"}


# ── Middleware (pure ASGI — não interfere no body stream de uploads) ──────────


class RequestIdMiddleware:
    """Pure ASGI middleware — does not wrap the request body stream like
    BaseHTTPMiddleware, so file uploads work correctly under uvicorn."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        token = request_id_var.set(request_id)

        async def send_with_request_id(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)


# ── App setup ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Agent Factory Lite starting up")
    yield
    logger.info("Agent Factory Lite shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error handling ───────────────────────────────────────────────────────────


def _format_internal_error(exc: Exception) -> str:
    if settings.environment.lower() == "development":
        detail = str(exc).strip()
        return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return "Internal Server Error"


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request_id_var.get()
    metrics.increment("http.unhandled_exceptions")
    logger.exception(
        "Unhandled application error",
        extra={
            "props": {
                "path": str(request.url.path),
                "method": request.method,
                "request_id": request_id,
            }
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": _format_internal_error(exc),
            "request_id": request_id,
        },
    )


# ── Schemas ──────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class IngestResponse(BaseModel):
    collection: str
    doc_id: str
    chunks_indexed: int


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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    request_id: str


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
    created_at: str
    updated_at: str


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


class AuditEventOut(BaseModel):
    id: int
    workspace_id: str
    actor: str
    event_type: str
    resource_type: str
    resource_id: str
    details: dict
    created_at: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sources_out(sources: list) -> list[SourceOut]:
    """Convert Source dataclass instances to SourceOut Pydantic models."""
    return [
        SourceOut(
            chunk_id=s.chunk_id,
            doc_id=s.doc_id,
            excerpt=s.excerpt,
            score=s.score,
            page_number=s.metadata.get("page_number") if s.metadata else None,
            source_filename=s.metadata.get("source_filename", "") if s.metadata else "",
            citation_label=_citation_label(s.metadata if s.metadata else {}),
        )
        for s in sources
    ]


def _map_stored_source(source: dict) -> SourceOut:
    metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
    return SourceOut(
        chunk_id=source.get("chunk_id", ""),
        doc_id=source.get("doc_id", ""),
        excerpt=source.get("excerpt", ""),
        score=source.get("score", 0.0),
        page_number=source.get("page_number") or metadata.get("page_number"),
        source_filename=source.get("source_filename", "") or metadata.get("source_filename", ""),
        citation_label=source.get("citation_label", "") or _citation_label(metadata),
    )


def _status_progress(status: str) -> tuple[str, int]:
    mapping = {
        "queued": ("bronze", 5),
        "bronze_received": ("bronze", 20),
        "silver_processing": ("silver", 45),
        "silver_extracted": ("silver", 65),
        "gold_indexing": ("gold", 85),
        "processing": ("gold", 80),
        "indexed": ("gold", 100),
        "failed": ("failed", 100),
    }
    return mapping.get(status, ("unknown", 0))


def _job_out(job) -> IngestionJobOut:
    stage, progress = _status_progress(job.status)
    return IngestionJobOut(**job.__dict__, stage=stage, progress_pct=progress)


def _candidate_artifact_keys(doc_id: str, filename: str) -> list[str]:
    candidates = [doc_id, pathlib.Path(doc_id).name, filename, pathlib.Path(filename).name, pathlib.Path(filename).stem]
    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = (item or "").strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _resolve_artifacts_for_document(doc_id: str, filename: str) -> tuple[pathlib.Path | None, pathlib.Path | None]:
    artifacts_dir = pathlib.Path(get_settings().pdf_pipeline_artifacts_dir).resolve()
    for key in _candidate_artifact_keys(doc_id, filename):
        md_path = (artifacts_dir / f"{key}.md").resolve()
        json_path = (artifacts_dir / f"{key}.json").resolve()
        if artifacts_dir not in md_path.parents or artifacts_dir not in json_path.parents:
            continue
        if md_path.exists() or json_path.exists():
            return (md_path if md_path.exists() else None, json_path if json_path.exists() else None)
    return None, None


def _citation_label(metadata: dict) -> str:
    if not metadata:
        return ""
    parts: list[str] = []
    if metadata.get("page_number") is not None:
        parts.append(f"pagina {metadata['page_number']}")
    for key in ("artigo", "paragrafo"):
        value = str(metadata.get(key, "")).strip()
        if value:
            parts.append(value)
    inciso = str(metadata.get("inciso", "")).strip()
    if inciso:
        parts.append(f"inciso {inciso}")
    caminho = str(metadata.get("caminho_hierarquico", "")).strip()
    if caminho:
        parts.append(caminho)
    return " | ".join(parts)


def _validate_collection_or_422(collection: str) -> str:
    try:
        return validate_collection(collection)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _validate_question_or_422(question: str) -> str:
    try:
        return sanitize_question(question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


def _run_answer_or_http_error(
    *,
    workspace_id: str,
    collection: str,
    question: str,
    history: list[ChatMessage],
    request_id: str,
    embedding_model: str,
    domain_profile: str | None,
    error_metric: str,
):
    try:
        return answer(
            workspace_id=workspace_id,
            collection=collection,
            question=question,
            history=history,
            request_id=request_id,
            embedding_model=embedding_model,
            domain_profile=domain_profile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        metrics.increment(error_metric)
        logger.exception(
            "Chat error",
            extra={
                "props": {
                    "collection": collection,
                    "embedding_model": embedding_model,
                    "domain_profile": domain_profile or "",
                    "request_id": request_id,
                }
            },
        )
        raise


def _validate_upload(file: UploadFile) -> None:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=422, detail=f"Unsupported file type. Allowed: {allowed}")

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 2 GB upload limit.")

    log_event(
        logger,
        20,
        "Upload validated",
        filename=file.filename or "",
        suffix=suffix,
        size_bytes=size,
    )


def _get_workspace(request: Request):
    api_key = request.headers.get("X-API-Key", "").strip() or None
    try:
        workspace = resolve_workspace(api_key, require_auth=settings.auth_required)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    return workspace


def _actor_from_request(request: Request) -> str:
    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key:
        return "api_key"
    return "anonymous"


def _process_ingestion_job(
    *,
    workspace_id: str,
    collection: str,
    doc_id: str,
    filename: str,
    embedding_model: str,
    domain_profile: str | None,
    tmp_path: str,
    job_id: str,
    actor: str,
) -> None:
    from src import ingestion

    def _run_ingest(stage_cb):
        kwargs = {
            "collection": collection,
            "source": tmp_path,
            "doc_id": doc_id,
            "embedding_model": embedding_model,
            "workspace_id": workspace_id,
            "domain_profile": domain_profile,
        }
        try:
            params = inspect.signature(ingestion.ingest).parameters
            if stage_cb and "stage_callback" in params:
                kwargs["stage_callback"] = stage_cb
        except (TypeError, ValueError):
            pass
        return ingestion.ingest(**kwargs)

    update_ingestion_job(job_id, status="bronze_received", started=True)
    upsert_document(
        workspace_id=workspace_id,
        collection=collection,
        doc_id=doc_id,
        filename=filename,
        embedding_model=embedding_model,
        status="bronze_received",
    )
    update_ingestion_job(job_id, status="silver_processing")
    upsert_document(
        workspace_id=workspace_id,
        collection=collection,
        doc_id=doc_id,
        filename=filename,
        embedding_model=embedding_model,
        status="silver_processing",
    )

    def _on_stage(status: str, _payload: dict) -> None:
        if status not in {"silver_extracted", "gold_indexing", "indexed"}:
            return
        chunks = _payload.get("chunks")
        update_ingestion_job(
            job_id,
            status=status,
            chunks_indexed=chunks if isinstance(chunks, int) else None,
        )
        upsert_document(
            workspace_id=workspace_id,
            collection=collection,
            doc_id=doc_id,
            filename=filename,
            embedding_model=embedding_model,
            status=status,
            chunks_indexed=chunks if isinstance(chunks, int) else 0,
            error="",
        )

    try:
        chunks = _run_ingest(_on_stage)
        update_ingestion_job(job_id, status="indexed", chunks_indexed=chunks, finished=True)
        upsert_document(
            workspace_id=workspace_id,
            collection=collection,
            doc_id=doc_id,
            filename=filename,
            embedding_model=embedding_model,
            status="indexed",
            chunks_indexed=chunks,
            error="",
        )
        record_audit(
            workspace_id=workspace_id,
            actor=actor,
            event_type="document.indexed",
            resource_type="document",
            resource_id=doc_id,
            details={"collection": collection, "embedding_model": embedding_model, "chunks_indexed": chunks},
        )
    except Exception as exc:
        update_ingestion_job(job_id, status="failed", error=str(exc), finished=True)
        upsert_document(
            workspace_id=workspace_id,
            collection=collection,
            doc_id=doc_id,
            filename=filename,
            embedding_model=embedding_model,
            status="failed",
            chunks_indexed=0,
            error=str(exc),
        )
        record_audit(
            workspace_id=workspace_id,
            actor=actor,
            event_type="document.failed",
            resource_type="document",
            resource_id=doc_id,
            details={"collection": collection, "embedding_model": embedding_model, "error": str(exc)},
        )
        logger.exception("Async ingest job failed", extra={"props": {"job_id": job_id, "workspace_id": workspace_id}})
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning("Could not delete temporary upload file", exc_info=True)


# ── Simple-mode UI ───────────────────────────────────────────────────────────

_TEMPLATES_DIR = pathlib.Path(__file__).resolve().parent / "templates"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def simple_ui():
    """Serve the built-in simple UI when simple_mode is enabled."""
    if not settings.simple_mode:
        return HTMLResponse("Simple mode is disabled. Use the React frontend.", status_code=404)
    html = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{ default_collection }}", settings.default_collection)
    return HTMLResponse(html)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )


class CollectionInfo(BaseModel):
    collection: str
    embedding_model: str


@app.get("/collections/available", response_model=list[CollectionInfo])
async def list_available_collections(request: Request):
    workspace = _get_workspace(request)
    return [
        CollectionInfo(**item)
        for item in list_collections_with_models(workspace.id)
    ]


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    collection: str = Form(...),
    embedding_model: str = Form(""),
    doc_id: str = Form(""),
    domain_profile: str = Form(""),
    file: UploadFile = File(...),
) -> IngestResponse:
    """Upload and index a document into a collection.

    Reads the uploaded file asynchronously via ``await file.read()`` so the
    body stream is fully consumed before any synchronous work begins.
    """
    workspace = _get_workspace(request)
    actor = _actor_from_request(request)
    client_ip = request.client.host if request.client else "unknown"
    if not ingest_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Limite de uploads excedido. Aguarde um momento.")
    collection = _validate_collection_or_422(collection)

    from src import ingestion

    with metrics.time_block("http.ingest.total"):
        _validate_upload(file)
        metrics.increment("ingest.requests")
        resolved_embedding_model = embedding_model or get_settings().embedding_model
        resolved_doc_id = doc_id or file.filename or ""
        log_event(
            logger,
            20,
            "Ingest request received",
            collection=collection,
            embedding_model=resolved_embedding_model,
            filename=file.filename or "",
            doc_id=resolved_doc_id,
        )

        # Lê o conteúdo via await — garante que o body stream ASGI é consumido
        # antes de passar para operações sync (embedding, ChromaDB).
        content = await file.read()

        suffix = "." + (file.filename or "file.txt").rsplit(".", 1)[-1]
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            upsert_document(
                workspace_id=workspace.id,
                collection=collection,
                doc_id=resolved_doc_id or tmp_path,
                filename=file.filename or resolved_doc_id or "",
                embedding_model=resolved_embedding_model,
                status="bronze_received",
                chunks_indexed=0,
                error="",
            )
            upsert_document(
                workspace_id=workspace.id,
                collection=collection,
                doc_id=resolved_doc_id or tmp_path,
                filename=file.filename or resolved_doc_id or "",
                embedding_model=resolved_embedding_model,
                status="silver_processing",
                chunks_indexed=0,
                error="",
            )

            def _on_stage(status: str, payload: dict) -> None:
                if status not in {"silver_extracted", "gold_indexing", "indexed"}:
                    return
                chunks = payload.get("chunks")
                upsert_document(
                    workspace_id=workspace.id,
                    collection=collection,
                    doc_id=resolved_doc_id or tmp_path,
                    filename=file.filename or resolved_doc_id or "",
                    embedding_model=resolved_embedding_model,
                    status=status,
                    chunks_indexed=chunks if isinstance(chunks, int) else 0,
                    error="",
                )

            def _run_ingest_with_optional_callback():
                kwargs = {
                    "collection": collection,
                    "source": tmp_path,
                    "doc_id": resolved_doc_id or tmp_path,
                    "embedding_model": resolved_embedding_model,
                    "workspace_id": workspace.id,
                    "domain_profile": domain_profile or None,
                }
                try:
                    params = inspect.signature(ingestion.ingest).parameters
                    if "stage_callback" in params:
                        kwargs["stage_callback"] = _on_stage
                except (TypeError, ValueError):
                    pass
                return ingestion.ingest(**kwargs)

            n = await asyncio.to_thread(
                _run_ingest_with_optional_callback,
            )
            upsert_document(
                workspace_id=workspace.id,
                collection=collection,
                doc_id=resolved_doc_id or tmp_path,
                filename=file.filename or resolved_doc_id or "",
                embedding_model=resolved_embedding_model,
                status="indexed",
                chunks_indexed=n,
                error="",
            )
            record_audit(
                workspace_id=workspace.id,
                actor=actor,
                event_type="document.indexed",
                resource_type="document",
                resource_id=resolved_doc_id or tmp_path,
                details={"collection": collection, "embedding_model": resolved_embedding_model, "chunks_indexed": n},
            )
        except HTTPException:
            metrics.increment("ingest.errors")
            raise
        except Exception as exc:
            metrics.increment("ingest.errors")
            upsert_document(
                workspace_id=workspace.id,
                collection=collection,
                doc_id=resolved_doc_id or file.filename or "",
                filename=file.filename or resolved_doc_id or "",
                embedding_model=resolved_embedding_model,
                status="failed",
                chunks_indexed=0,
                error=str(exc),
            )
            record_audit(
                workspace_id=workspace.id,
                actor=actor,
                event_type="document.failed",
                resource_type="document",
                resource_id=resolved_doc_id or file.filename or "",
                details={"collection": collection, "embedding_model": resolved_embedding_model, "error": str(exc)},
            )
            logger.exception(
                "Ingest error",
                extra={
                    "props": {
                        "collection": collection,
                        "embedding_model": resolved_embedding_model,
                        "filename": file.filename or "",
                        "doc_id": resolved_doc_id,
                    }
                },
            )
            raise HTTPException(status_code=422, detail=str(exc))
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.warning("Could not delete temporary upload file", exc_info=True)

    return IngestResponse(
        collection=collection,
        doc_id=doc_id or file.filename or "",
        chunks_indexed=n,
    )



@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, request: Request) -> ChatResponse:
    workspace = _get_workspace(request)
    client_ip = request.client.host if request.client else "unknown"
    if not chat_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Limite de requisições excedido. Aguarde um momento.")
    req.collection = _validate_collection_or_422(req.collection)
    req.question = _validate_question_or_422(req.question)

    request_id = request_id_var.get()
    with metrics.time_block("http.chat.total"):
        metrics.increment("chat.requests")
        log_event(
            logger,
            20,
            "Chat request received",
            collection=req.collection,
            embedding_model=req.embedding_model or get_settings().embedding_model,
            domain_profile=req.domain_profile or get_settings().default_domain_profile,
            history_size=len(req.history),
            request_id=request_id,
        )
        result = _run_answer_or_http_error(
            workspace_id=workspace.id,
            collection=req.collection,
            question=req.question,
            history=req.history,
            request_id=request_id,
            embedding_model=req.embedding_model or get_settings().embedding_model,
            domain_profile=req.domain_profile or get_settings().default_domain_profile,
            error_metric="chat.errors",
        )

    return ChatResponse(
        answer=result.answer,
        sources=_sources_out(result.sources),
        request_id=result.request_id,
    )


@app.get("/conversations", response_model=list[ConversationOut])
def list_convs(request: Request, q: str = Query(default="", alias="q")):
    from src.history import list_conversations, search_conversations

    workspace = _get_workspace(request)
    convs = (
        search_conversations(workspace.id, q, limit=30)
        if q.strip()
        else list_conversations(workspace.id, limit=40)
    )
    return [ConversationOut(**c.__dict__) for c in convs]


@app.post("/conversations", response_model=ConversationOut)
def create_conv(req: CreateConversationRequest, request: Request):
    from src.history import create_conversation, get_conversation

    workspace = _get_workspace(request)
    req.collection = _validate_collection_or_422(req.collection)
    conv_id = create_conversation(
        workspace_id=workspace.id,
        collection=req.collection,
        embedding_model=req.embedding_model or get_settings().embedding_model,
        title=req.title,
    )
    conv = get_conversation(conv_id, workspace.id)
    return ConversationOut(**conv.__dict__)



@app.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
def get_messages(conv_id: str, request: Request):
    from src.history import get_conversation, load_messages

    workspace = _get_workspace(request)
    conv = get_conversation(conv_id, workspace.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = load_messages(conv_id)
    return [
        MessageOut(
            role=m.role,
            content=m.content,
            sources=[_map_stored_source(s) for s in m.sources],
        )
        for m in msgs
    ]


@app.patch("/conversations/{conv_id}", response_model=ConversationOut)
def rename_conv(conv_id: str, req: RenameRequest, request: Request):
    from src.history import rename_conversation, get_conversation

    workspace = _get_workspace(request)
    conv = get_conversation(conv_id, workspace.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    rename_conversation(conv_id, req.title)
    conv = get_conversation(conv_id, workspace.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationOut(**conv.__dict__)


@app.delete("/conversations/{conv_id}", status_code=204)
def delete_conv(conv_id: str, request: Request):
    from src.history import delete_conversation, get_conversation

    workspace = _get_workspace(request)
    conv = get_conversation(conv_id, workspace.id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    delete_conversation(conv_id)


@app.post("/chat/message", response_model=ChatWithHistoryResponse)
def chat_message(req: ChatWithHistoryRequest, request: Request) -> ChatWithHistoryResponse:
    workspace = _get_workspace(request)
    actor = _actor_from_request(request)
    client_ip = request.client.host if request.client else "unknown"
    if not chat_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Limite de requisições excedido. Aguarde um momento.")
    req.collection = _validate_collection_or_422(req.collection)
    req.question = _validate_question_or_422(req.question)

    from src.history import create_conversation, save_message, rename_conversation, get_conversation

    request_id = request_id_var.get()
    with metrics.time_block("http.chat_message.total"):
        metrics.increment("chat_message.requests")
        log_event(
            logger,
            20,
            "Chat message request received",
            collection=req.collection,
            conversation_id=req.conversation_id or "",
            requested_embedding_model=req.embedding_model or "",
            domain_profile=req.domain_profile or get_settings().default_domain_profile,
            history_size=len(req.history),
            request_id=request_id,
        )

        conv = get_conversation(req.conversation_id, workspace.id) if req.conversation_id else None
        embedding_model = (
            conv.embedding_model
            if conv and conv.embedding_model
            else req.embedding_model or get_settings().embedding_model
        )

        if conv:
            conv_id = conv.id
        else:
            conv_id = create_conversation(
                workspace_id=workspace.id,
                collection=req.collection,
                embedding_model=embedding_model,
            )
            conv = get_conversation(conv_id, workspace.id)
            log_event(
                logger,
                20,
                "Conversation created for chat message",
                conversation_id=conv_id,
                collection=req.collection,
                embedding_model=embedding_model,
            )

        if conv and conv.title.startswith("Conversa "):
            rename_conversation(conv_id, req.question[:60])

        save_message(conv_id, "user", req.question)

        result = _run_answer_or_http_error(
            workspace_id=workspace.id,
            collection=req.collection,
            question=req.question,
            history=req.history,
            request_id=request_id,
            embedding_model=embedding_model,
            domain_profile=req.domain_profile or get_settings().default_domain_profile,
            error_metric="chat_message.errors",
        )

        save_message(conv_id, "assistant", result.answer, result.sources)
        record_audit(
            workspace_id=workspace.id,
            actor=actor,
            event_type="chat.completed",
            resource_type="conversation",
            resource_id=conv_id,
            details={"collection": req.collection, "sources": len(result.sources)},
        )

    return ChatWithHistoryResponse(
        conversation_id=conv_id,
        answer=result.answer,
        sources=_sources_out(result.sources),
        request_id=result.request_id,
    )


# ── Streaming chat ────────────────────────────────────────────────────────────


@app.post("/chat/message/stream")
async def chat_message_stream(req: ChatWithHistoryRequest, request: Request):
    """SSE endpoint — streams LLM tokens as they arrive."""
    import json as _json

    from starlette.responses import StreamingResponse

    from src import llm
    from src.history import create_conversation, save_message, rename_conversation, get_conversation

    workspace = _get_workspace(request)
    actor = _actor_from_request(request)
    client_ip = request.client.host if request.client else "unknown"
    if not chat_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Limite de requisições excedido. Aguarde um momento.")
    req.collection = _validate_collection_or_422(req.collection)
    req.question = _validate_question_or_422(req.question)

    request_id = request_id_var.get()
    embedding_model = req.embedding_model or get_settings().embedding_model

    conv = get_conversation(req.conversation_id, workspace.id) if req.conversation_id else None
    if conv and conv.embedding_model:
        embedding_model = conv.embedding_model

    if conv:
        conv_id = conv.id
    else:
        conv_id = create_conversation(
            workspace_id=workspace.id,
            collection=req.collection,
            embedding_model=embedding_model,
        )
        conv = get_conversation(conv_id, workspace.id)

    if conv and conv.title.startswith("Conversa "):
        rename_conversation(conv_id, req.question[:60])

    save_message(conv_id, "user", req.question)

    # Run retrieval synchronously (fast), then stream LLM
    ctx = await asyncio.to_thread(
        prepare_stream,
        collection=req.collection,
        question=req.question,
        history=req.history,
        request_id=request_id,
        embedding_model=embedding_model,
        workspace_id=workspace.id,
        domain_profile=req.domain_profile or get_settings().default_domain_profile,
    )

    if isinstance(ctx, ChatResult):
        # Blocked or empty — return as a single SSE event
        save_message(conv_id, "assistant", ctx.answer, ctx.sources)

        async def _blocked():
            data = _json.dumps({
                "type": "sources",
                "sources": [],
                "conversation_id": conv_id,
                "request_id": request_id,
            })
            yield f"data: {data}\n\n"
            yield f"data: {_json.dumps({'type': 'token', 'token': ctx.answer})}\n\n"
            yield f"data: {_json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(_blocked(), media_type="text/event-stream")

    # Stream LLM tokens
    async def _generate():
        # First event: sources metadata
        sources_data = _json.dumps({
            "type": "sources",
            "sources": [
                {
                    "chunk_id": s.chunk_id,
                    "doc_id": s.doc_id,
                    "excerpt": s.excerpt,
                    "score": s.score,
                    "page_number": s.metadata.get("page_number") if s.metadata else None,
                    "source_filename": s.metadata.get("source_filename", "") if s.metadata else "",
                    "citation_label": _citation_label(s.metadata if s.metadata else {}),
                }
                for s in ctx.sources
            ],
            "conversation_id": conv_id,
            "request_id": request_id,
        })
        yield f"data: {sources_data}\n\n"

        # Stream tokens from LLM
        full_answer: list[str] = []

        def _stream_sync():
            for token in llm.chat_stream(ctx.messages, system=ctx.system):
                full_answer.append(token)
                yield token

        # Run the sync generator in a thread, forwarding tokens
        import queue
        import threading

        q: queue.Queue[str | None] = queue.Queue()

        def _worker():
            try:
                for token in llm.chat_stream(ctx.messages, system=ctx.system):
                    q.put(token)
            except Exception as exc:
                q.put(None)
                raise exc
            finally:
                q.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        while True:
            token = await asyncio.to_thread(q.get)
            if token is None:
                break
            full_answer.append(token)
            yield f"data: {_json.dumps({'type': 'token', 'token': token})}\n\n"

        thread.join(timeout=5)

        final_answer = "".join(full_answer)
        save_message(conv_id, "assistant", final_answer, ctx.sources)
        record_audit(
            workspace_id=workspace.id,
            actor=actor,
            event_type="chat.completed",
            resource_type="conversation",
            resource_id=conv_id,
            details={"collection": req.collection, "sources": len(ctx.sources)},
        )

        yield f"data: {_json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── Documents (available in both modes for the simple UI doc list) ────────────


@app.get("/documents", response_model=list[DocumentOut])
def get_documents(request: Request, collection: str | None = Query(default=None)):
    workspace = _get_workspace(request)
    if collection is not None:
        collection = _validate_collection_or_422(collection)
    docs = list_document_records(workspace.id, collection=collection)
    return [DocumentOut(**doc.__dict__) for doc in docs]


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, request: Request, collection: str = Query(...), embedding_model: str | None = Query(default=None)):
    from src import chat, vectordb

    workspace = _get_workspace(request)
    collection = _validate_collection_or_422(collection)
    docs = [get_document(workspace.id, collection, doc_id, embedding_model)] if embedding_model else [
        doc for doc in list_document_records(workspace.id, collection=collection) if doc.doc_id == doc_id
    ]
    docs = [doc for doc in docs if doc is not None]
    if not docs:
        raise HTTPException(status_code=404, detail="Document not found")

    deleted_chunks = 0
    for doc in docs:
        physical_collection = vectordb.collection_key(
            collection,
            doc.embedding_model,
            workspace_id=workspace.id,
        )
        deleted_chunks += vectordb.delete_by_doc_id(physical_collection, doc_id)
    delete_document_record(workspace.id, collection, doc_id, embedding_model)
    record_audit(
        workspace_id=workspace.id,
        actor=_actor_from_request(request),
        event_type="document.deleted",
        resource_type="document",
        resource_id=doc_id,
        details={"collection": collection, "deleted_chunks": deleted_chunks},
    )


@app.get("/documents/{doc_id}/artifacts", response_model=DocumentArtifactsOut)
def get_document_artifacts(
    doc_id: str,
    request: Request,
    collection: str = Query(...),
    embedding_model: str | None = Query(default=None),
):
    workspace = _get_workspace(request)
    collection = _validate_collection_or_422(collection)

    doc = get_document(workspace.id, collection, doc_id, embedding_model)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    md_path, js_path = _resolve_artifacts_for_document(doc.doc_id, doc.filename)
    if not md_path and not js_path:
        raise HTTPException(status_code=404, detail="Silver artifacts not found")

    md_preview = md_path.read_text(encoding="utf-8", errors="ignore")[:50000] if md_path else ""
    if js_path:
        try:
            payload = json.loads(js_path.read_text(encoding="utf-8", errors="ignore"))
            js_preview = json.dumps(payload, ensure_ascii=False, indent=2)[:50000]
        except Exception:
            js_preview = js_path.read_text(encoding="utf-8", errors="ignore")[:50000]
    else:
        js_preview = ""

    return DocumentArtifactsOut(
        doc_id=doc.doc_id,
        markdown_path=str(md_path) if md_path else "",
        json_path=str(js_path) if js_path else "",
        markdown_preview=md_preview,
        json_preview=js_preview,
        available=bool(md_path or js_path),
    )


@app.get("/documents/{doc_id}/artifacts/download")
def download_document_artifact(
    doc_id: str,
    request: Request,
    collection: str = Query(...),
    kind: str = Query(..., pattern="^(markdown|json)$"),
    embedding_model: str | None = Query(default=None),
):
    workspace = _get_workspace(request)
    collection = _validate_collection_or_422(collection)
    doc = get_document(workspace.id, collection, doc_id, embedding_model)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    md_path, js_path = _resolve_artifacts_for_document(doc.doc_id, doc.filename)
    target = md_path if kind == "markdown" else js_path
    if not target:
        raise HTTPException(status_code=404, detail="Artifact not found")

    media_type = "text/markdown; charset=utf-8" if kind == "markdown" else "application/json"
    out_name = target.name
    return FileResponse(path=str(target), media_type=media_type, filename=out_name)


# ── Enterprise routes (hidden when simple_mode=True) ─────────────────────────

enterprise_router = APIRouter()


@enterprise_router.get("/settings", response_model=SettingsOut)
async def get_current_settings():
    cfg = get_settings()
    return SettingsOut(
        llm_provider=cfg.llm_provider,
        llm_model=cfg.llm_model,
        embedding_model=cfg.embedding_model,
        retrieval_top_k=cfg.retrieval_top_k,
        default_domain_profile=cfg.default_domain_profile,
        available_domain_profiles=list_domain_profiles(),
    )


@enterprise_router.get("/evaluation/retrieval", response_model=RetrievalEvaluationOut)
async def get_retrieval_evaluation(top_k: int = Query(default=5, ge=1, le=10)):
    return RetrievalEvaluationOut(**evaluate_retrieval_snapshot(top_k=top_k))


@enterprise_router.get("/observability", response_model=ObservabilityOut)
async def get_observability():
    return ObservabilityOut(**metrics.snapshot())


@enterprise_router.get("/workspaces", response_model=list[WorkspaceOut])
def get_workspaces():
    ensure_default_workspace()
    return [WorkspaceOut(**workspace.__dict__) for workspace in list_workspaces()]


@enterprise_router.post("/workspaces", response_model=WorkspaceOut)
def post_workspace(req: CreateWorkspaceRequest):
    workspace = create_workspace(req.name.strip())
    return WorkspaceOut(**workspace.__dict__)


@enterprise_router.get("/collections", response_model=list[CollectionStatsOut])
def get_collection_stats(request: Request):
    workspace = _get_workspace(request)
    return [CollectionStatsOut(**item) for item in list_collection_stats(workspace.id)]


@enterprise_router.post("/ingest/async", response_model=IngestionJobOut)
async def ingest_document_async(
    background_tasks: BackgroundTasks,
    request: Request,
    collection: str = Form(...),
    embedding_model: str = Form(""),
    doc_id: str = Form(""),
    domain_profile: str = Form(""),
    file: UploadFile = File(...),
) -> IngestionJobOut:
    workspace = _get_workspace(request)
    actor = _actor_from_request(request)
    client_ip = request.client.host if request.client else "unknown"
    if not ingest_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Limite de uploads excedido. Aguarde um momento.")
    collection = _validate_collection_or_422(collection)
    _validate_upload(file)

    resolved_embedding_model = embedding_model or get_settings().embedding_model
    resolved_doc_id = doc_id or file.filename or ""
    content = await file.read()
    suffix = "." + (file.filename or "file.txt").rsplit(".", 1)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    job = create_ingestion_job(
        workspace_id=workspace.id,
        collection=collection,
        doc_id=resolved_doc_id or tmp_path,
        filename=file.filename or resolved_doc_id or "",
        embedding_model=resolved_embedding_model,
    )
    upsert_document(
        workspace_id=workspace.id,
        collection=collection,
        doc_id=resolved_doc_id or tmp_path,
        filename=file.filename or resolved_doc_id or "",
        embedding_model=resolved_embedding_model,
        status="bronze_received",
    )
    record_audit(
        workspace_id=workspace.id,
        actor=actor,
        event_type="document.queued",
        resource_type="ingestion_job",
        resource_id=job.id,
        details={"collection": collection, "doc_id": resolved_doc_id or tmp_path},
    )
    background_tasks.add_task(
        _process_ingestion_job,
        workspace_id=workspace.id,
        collection=collection,
        doc_id=resolved_doc_id or tmp_path,
        filename=file.filename or resolved_doc_id or "",
        embedding_model=resolved_embedding_model,
        domain_profile=domain_profile or None,
        tmp_path=tmp_path,
        job_id=job.id,
        actor=actor,
    )
    return _job_out(job)


@enterprise_router.get("/ingest/jobs", response_model=list[IngestionJobOut])
def get_ingest_jobs(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    workspace = _get_workspace(request)
    return [_job_out(job) for job in list_ingestion_jobs(workspace.id, limit=limit)]


@enterprise_router.get("/ingest/jobs/{job_id}", response_model=IngestionJobOut)
def get_ingest_job(job_id: str, request: Request):
    workspace = _get_workspace(request)
    job = get_ingestion_job(job_id, workspace.id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return _job_out(job)


@enterprise_router.get("/audit/events", response_model=list[AuditEventOut])
def get_audit_events(request: Request, limit: int = Query(default=100, ge=1, le=500)):
    workspace = _get_workspace(request)
    return [AuditEventOut(**event.__dict__) for event in list_audit_events(workspace.id, limit=limit)]


@enterprise_router.delete("/admin/purge-conversations")
def purge_conversations_enterprise(older_than_days: int = Query(default=90, ge=1)):
    """Delete conversations not updated within the given number of days."""
    from src.history import purge_old_conversations

    deleted = purge_old_conversations(older_than_days)
    return {"deleted": deleted, "older_than_days": older_than_days}


if not settings.simple_mode:
    app.include_router(enterprise_router)
