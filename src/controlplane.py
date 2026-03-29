"""Workspace, documents, ingestion jobs and audit trail control plane."""
from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings
from src.utils import get_logger, log_event

_DB_PATH = Path("data/history.db")
logger = get_logger(__name__)


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@dataclass
class Workspace:
    id: str
    name: str
    api_key: str
    is_default: bool
    created_at: str


@dataclass
class DocumentRecord:
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


@dataclass
class IngestionJob:
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


@dataclass
class AuditEvent:
    id: int
    workspace_id: str
    actor: str
    event_type: str
    resource_type: str
    resource_id: str
    details: dict
    created_at: str


def init_db() -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL UNIQUE,
                    api_key     TEXT NOT NULL UNIQUE,
                    is_default  INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id              TEXT PRIMARY KEY,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    doc_id          TEXT NOT NULL,
                    filename        TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    chunks_indexed  INTEGER NOT NULL DEFAULT 0,
                    error           TEXT NOT NULL DEFAULT '',
                    context_hint    TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE(workspace_id, collection, doc_id, embedding_model),
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id              TEXT PRIMARY KEY,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    doc_id          TEXT NOT NULL,
                    filename        TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL,
                    status          TEXT NOT NULL,
                    chunks_indexed  INTEGER NOT NULL DEFAULT 0,
                    error           TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    started_at      TEXT NOT NULL DEFAULT '',
                    finished_at     TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS audit_events (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id  TEXT NOT NULL,
                    actor         TEXT NOT NULL DEFAULT 'system',
                    event_type    TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id   TEXT NOT NULL DEFAULT '',
                    details_json  TEXT NOT NULL DEFAULT '{}',
                    created_at    TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS document_nodes (
                    id              TEXT PRIMARY KEY,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    doc_id          TEXT NOT NULL,
                    node_type       TEXT NOT NULL,
                    label           TEXT NOT NULL DEFAULT '',
                    numeral         TEXT NOT NULL DEFAULT '',
                    path            TEXT NOT NULL DEFAULT '',
                    parent_node_id  TEXT NOT NULL DEFAULT '',
                    text_length     INTEGER NOT NULL DEFAULT 0,
                    articles_json   TEXT NOT NULL DEFAULT '[]',
                    refs_json       TEXT NOT NULL DEFAULT '[]',
                    tree_json       TEXT NOT NULL DEFAULT '{}',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE(workspace_id, collection, doc_id, node_type, label)
                );

                CREATE TABLE IF NOT EXISTS document_summaries (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    doc_id          TEXT NOT NULL,
                    node_id         TEXT NOT NULL,
                    node_type       TEXT NOT NULL,
                    label           TEXT NOT NULL DEFAULT '',
                    path            TEXT NOT NULL DEFAULT '',
                    resumo_executivo    TEXT NOT NULL DEFAULT '',
                    resumo_juridico     TEXT NOT NULL DEFAULT '',
                    pontos_chave_json   TEXT NOT NULL DEFAULT '[]',
                    artigos_cobertos_json TEXT NOT NULL DEFAULT '[]',
                    obrigacoes_json     TEXT NOT NULL DEFAULT '[]',
                    restricoes_json     TEXT NOT NULL DEFAULT '[]',
                    definicoes_json     TEXT NOT NULL DEFAULT '[]',
                    text_length     INTEGER NOT NULL DEFAULT 0,
                    source_hash     TEXT NOT NULL DEFAULT '',
                    source_text_length INTEGER NOT NULL DEFAULT 0,
                    status          TEXT NOT NULL DEFAULT 'generated',
                    invalid_reason  TEXT NOT NULL DEFAULT '',
                    generation_meta_json TEXT NOT NULL DEFAULT '{}',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE(workspace_id, collection, doc_id, node_id)
                );

                CREATE INDEX IF NOT EXISTS idx_documents_workspace_collection
                    ON documents(workspace_id, collection, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_jobs_workspace_created
                    ON ingestion_jobs(workspace_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_audit_workspace_created
                    ON audit_events(workspace_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_nodes_workspace_collection
                    ON document_nodes(workspace_id, collection, doc_id);
                CREATE INDEX IF NOT EXISTS idx_nodes_type
                    ON document_nodes(workspace_id, collection, node_type);
                CREATE INDEX IF NOT EXISTS idx_summaries_workspace_collection
                    ON document_summaries(workspace_id, collection, doc_id);
                CREATE INDEX IF NOT EXISTS idx_summaries_node
                    ON document_summaries(workspace_id, collection, node_id);

                CREATE TABLE IF NOT EXISTS table_profiles (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    table_name      TEXT NOT NULL DEFAULT '',
                    base_context    TEXT NOT NULL DEFAULT '',
                    subject_label   TEXT NOT NULL DEFAULT '',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE(workspace_id, collection)
                );

                CREATE TABLE IF NOT EXISTS column_profiles (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    column_name     TEXT NOT NULL,
                    display_name    TEXT NOT NULL DEFAULT '',
                    physical_type   TEXT NOT NULL DEFAULT '',
                    semantic_type   TEXT NOT NULL DEFAULT '',
                    role            TEXT NOT NULL DEFAULT '',
                    unit            TEXT NOT NULL DEFAULT '',
                    aliases_json    TEXT NOT NULL DEFAULT '[]',
                    examples_json   TEXT NOT NULL DEFAULT '[]',
                    description     TEXT NOT NULL DEFAULT '',
                    cardinality     INTEGER NOT NULL DEFAULT 0,
                    allowed_ops_json TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    UNIQUE(workspace_id, collection, column_name)
                );

                CREATE TABLE IF NOT EXISTS value_catalog (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id    TEXT NOT NULL,
                    collection      TEXT NOT NULL,
                    column_name     TEXT NOT NULL,
                    normalized_value TEXT NOT NULL DEFAULT '',
                    raw_value       TEXT NOT NULL DEFAULT '',
                    frequency       INTEGER NOT NULL DEFAULT 0,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    updated_at      TEXT NOT NULL DEFAULT (datetime('now','localtime'))
                );

                CREATE TABLE IF NOT EXISTS query_plans_log (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id        TEXT NOT NULL,
                    collection          TEXT NOT NULL,
                    question            TEXT NOT NULL DEFAULT '',
                    planner_source      TEXT NOT NULL DEFAULT '',
                    plan_json           TEXT NOT NULL DEFAULT '{}',
                    validated           INTEGER NOT NULL DEFAULT 0,
                    validation_errors_json TEXT NOT NULL DEFAULT '[]',
                    sql_generated       TEXT NOT NULL DEFAULT '',
                    created_at          TEXT NOT NULL DEFAULT (datetime('now','localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_table_profiles_workspace_collection
                    ON table_profiles(workspace_id, collection);
                CREATE INDEX IF NOT EXISTS idx_column_profiles_workspace_collection
                    ON column_profiles(workspace_id, collection);
                CREATE INDEX IF NOT EXISTS idx_value_catalog_workspace_collection
                    ON value_catalog(workspace_id, collection, column_name);
                CREATE INDEX IF NOT EXISTS idx_query_plans_workspace_collection
                    ON query_plans_log(workspace_id, collection, created_at DESC);
                """
            )

    _ensure_schema_migrations()
    ensure_default_workspace()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _ensure_schema_migrations() -> None:
    """Apply additive migrations for existing local SQLite databases."""
    with closing(_connect()) as conn:
        with conn:
            additions = [
                ("documents", "context_hint", "TEXT NOT NULL DEFAULT ''"),
                ("document_summaries", "source_hash", "TEXT NOT NULL DEFAULT ''"),
                ("document_summaries", "source_text_length", "INTEGER NOT NULL DEFAULT 0"),
                ("document_summaries", "status", "TEXT NOT NULL DEFAULT 'generated'"),
                ("document_summaries", "invalid_reason", "TEXT NOT NULL DEFAULT ''"),
                ("document_summaries", "generation_meta_json", "TEXT NOT NULL DEFAULT '{}'"),
            ]
            for table, column, ddl in additions:
                if not _column_exists(conn, table, column):
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _row_to_workspace(row: sqlite3.Row) -> Workspace:
    return Workspace(
        id=row["id"],
        name=row["name"],
        api_key=row["api_key"],
        is_default=bool(row["is_default"]),
        created_at=row["created_at"],
    )


def _row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        workspace_id=row["workspace_id"],
        collection=row["collection"],
        doc_id=row["doc_id"],
        filename=row["filename"],
        embedding_model=row["embedding_model"],
        status=row["status"],
        chunks_indexed=row["chunks_indexed"],
        error=row["error"],
        context_hint=row["context_hint"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_job(row: sqlite3.Row) -> IngestionJob:
    return IngestionJob(
        id=row["id"],
        workspace_id=row["workspace_id"],
        collection=row["collection"],
        doc_id=row["doc_id"],
        filename=row["filename"],
        embedding_model=row["embedding_model"],
        status=row["status"],
        chunks_indexed=row["chunks_indexed"],
        error=row["error"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )


def _row_to_audit(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        id=row["id"],
        workspace_id=row["workspace_id"],
        actor=row["actor"],
        event_type=row["event_type"],
        resource_type=row["resource_type"],
        resource_id=row["resource_id"],
        details=json.loads(row["details_json"]),
        created_at=row["created_at"],
    )


def ensure_default_workspace() -> Workspace:
    default_name = get_settings().default_workspace_name
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM workspaces WHERE is_default=1 ORDER BY created_at LIMIT 1"
        ).fetchone()
        if row:
            return _row_to_workspace(row)

        workspace_id = "default"
        api_key = f"af_{secrets.token_urlsafe(24)}"
        with conn:
            conn.execute(
                "INSERT OR IGNORE INTO workspaces (id, name, api_key, is_default) VALUES (?,?,?,1)",
                (workspace_id, default_name, api_key),
            )
        row = conn.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
        assert row is not None
        log_event(logger, 20, "Default workspace ensured", workspace_id=workspace_id, workspace_name=default_name)
        return _row_to_workspace(row)


def list_workspaces() -> list[Workspace]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM workspaces ORDER BY created_at").fetchall()
    return [_row_to_workspace(row) for row in rows]


def create_workspace(name: str) -> Workspace:
    workspace_id = str(uuid.uuid4())
    api_key = f"af_{secrets.token_urlsafe(24)}"
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "INSERT INTO workspaces (id, name, api_key, is_default) VALUES (?,?,?,0)",
                (workspace_id, name, api_key),
            )
        row = conn.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
    assert row is not None
    log_event(logger, 20, "Workspace created", workspace_id=workspace_id, workspace_name=name)
    record_audit(
        workspace_id=workspace_id,
        actor="system",
        event_type="workspace.created",
        resource_type="workspace",
        resource_id=workspace_id,
        details={"name": name},
    )
    return _row_to_workspace(row)


def get_workspace_by_api_key(api_key: str) -> Workspace | None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM workspaces WHERE api_key=?", (api_key,)).fetchone()
    return _row_to_workspace(row) if row else None


def get_workspace(workspace_id: str) -> Workspace | None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
    return _row_to_workspace(row) if row else None


def resolve_workspace(api_key: str | None, require_auth: bool) -> Workspace:
    if api_key:
        workspace = get_workspace_by_api_key(api_key)
        if workspace:
            return workspace
        if require_auth:
            raise ValueError("Invalid API key.")

    if require_auth:
        raise ValueError("API key required.")
    return ensure_default_workspace()


def upsert_document(
    *,
    workspace_id: str,
    collection: str,
    doc_id: str,
    filename: str,
    embedding_model: str,
    status: str,
    chunks_indexed: int = 0,
    error: str = "",
    context_hint: str = "",
) -> DocumentRecord:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, workspace_id, collection, doc_id, filename, embedding_model,
                    status, chunks_indexed, error, context_hint
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(workspace_id, collection, doc_id, embedding_model)
                DO UPDATE SET
                    filename=excluded.filename,
                    status=excluded.status,
                    chunks_indexed=excluded.chunks_indexed,
                    error=excluded.error,
                    context_hint=CASE
                        WHEN excluded.context_hint <> '' THEN excluded.context_hint
                        ELSE documents.context_hint
                    END,
                    updated_at=datetime('now','localtime')
                """,
                (
                    str(uuid.uuid4()),
                    workspace_id,
                    collection,
                    doc_id,
                    filename,
                    embedding_model,
                    status,
                    chunks_indexed,
                    error,
                    context_hint,
                ),
            )
        row = conn.execute(
            """
            SELECT * FROM documents
            WHERE workspace_id=? AND collection=? AND doc_id=? AND embedding_model=?
            """,
            (workspace_id, collection, doc_id, embedding_model),
        ).fetchone()
    assert row is not None
    return _row_to_document(row)


def list_documents(workspace_id: str, collection: str | None = None) -> list[DocumentRecord]:
    query = "SELECT * FROM documents WHERE workspace_id=?"
    params: list[str] = [workspace_id]
    if collection:
        query += " AND collection=?"
        params.append(collection)
    query += " ORDER BY updated_at DESC"
    with closing(_connect()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_document(row) for row in rows]


def get_document(workspace_id: str, collection: str, doc_id: str, embedding_model: str | None = None) -> DocumentRecord | None:
    query = "SELECT * FROM documents WHERE workspace_id=? AND collection=? AND doc_id=?"
    params: list[str] = [workspace_id, collection, doc_id]
    if embedding_model:
        query += " AND embedding_model=?"
        params.append(embedding_model)
    query += " ORDER BY updated_at DESC LIMIT 1"
    with closing(_connect()) as conn:
        row = conn.execute(query, params).fetchone()
    return _row_to_document(row) if row else None


def delete_document_record(workspace_id: str, collection: str, doc_id: str, embedding_model: str | None = None) -> int:
    query = "DELETE FROM documents WHERE workspace_id=? AND collection=? AND doc_id=?"
    params: list[str] = [workspace_id, collection, doc_id]
    if embedding_model:
        query += " AND embedding_model=?"
        params.append(embedding_model)
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount


def get_collection_context(workspace_id: str, collection: str) -> str:
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT context_hint
            FROM documents
            WHERE workspace_id=? AND collection=? AND TRIM(context_hint) <> ''
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (workspace_id, collection),
        ).fetchone()
    return str(row["context_hint"]).strip() if row else ""


def update_collection_context(workspace_id: str, collection: str, context_hint: str) -> int:
    cleaned = (context_hint or "").strip()
    from src.table_semantics import infer_subject_label

    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                """
                UPDATE documents
                SET context_hint=?,
                    updated_at=datetime('now','localtime')
                WHERE workspace_id=? AND collection=?
                """,
                (cleaned, workspace_id, collection),
            )
            conn.execute(
                """
                INSERT INTO table_profiles (
                    workspace_id, collection, table_name, base_context, subject_label
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(workspace_id, collection)
                DO UPDATE SET
                    base_context=excluded.base_context,
                    subject_label=excluded.subject_label,
                    updated_at=datetime('now','localtime')
                """,
                (
                    workspace_id,
                    collection,
                    "",
                    cleaned,
                    infer_subject_label("", cleaned),
                ),
            )
            return cursor.rowcount


def upsert_table_profile(
    workspace_id: str,
    collection: str,
    table_name: str,
    base_context: str,
    subject_label: str,
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO table_profiles (
                    workspace_id, collection, table_name, base_context, subject_label
                ) VALUES (?,?,?,?,?)
                ON CONFLICT(workspace_id, collection)
                DO UPDATE SET
                    table_name=excluded.table_name,
                    base_context=excluded.base_context,
                    subject_label=excluded.subject_label,
                    updated_at=datetime('now','localtime')
                """,
                (workspace_id, collection, table_name, base_context, subject_label),
            )


def get_table_profile(workspace_id: str, collection: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            """
            SELECT workspace_id, collection, table_name, base_context, subject_label,
                   created_at, updated_at
            FROM table_profiles
            WHERE workspace_id=? AND collection=?
            LIMIT 1
            """,
            (workspace_id, collection),
        ).fetchone()
    return dict(row) if row else None


def replace_column_profiles(workspace_id: str, collection: str, profiles: list[dict]) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM column_profiles WHERE workspace_id=? AND collection=?",
                (workspace_id, collection),
            )
            for profile in profiles:
                conn.execute(
                    """
                    INSERT INTO column_profiles (
                        workspace_id, collection, column_name, display_name, physical_type,
                        semantic_type, role, unit, aliases_json, examples_json, description,
                        cardinality, allowed_ops_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        workspace_id,
                        collection,
                        profile.get("name", ""),
                        profile.get("display_name") or profile.get("name", ""),
                        profile.get("data_type", ""),
                        profile.get("semantic_type", ""),
                        profile.get("role", ""),
                        profile.get("unit", ""),
                        json.dumps(profile.get("aliases", []), ensure_ascii=False),
                        json.dumps(profile.get("examples", []), ensure_ascii=False),
                        profile.get("description", ""),
                        int(profile.get("cardinality", 0) or 0),
                        json.dumps(profile.get("allowed_operations", []), ensure_ascii=False),
                    ),
                )


def list_column_profiles(workspace_id: str, collection: str) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT * FROM column_profiles
            WHERE workspace_id=? AND collection=?
            ORDER BY id ASC
            """,
            (workspace_id, collection),
        ).fetchall()
    results: list[dict] = []
    for row in rows:
        item = dict(row)
        item["aliases"] = json.loads(item.pop("aliases_json", "[]") or "[]")
        item["examples"] = json.loads(item.pop("examples_json", "[]") or "[]")
        item["allowed_operations"] = json.loads(item.pop("allowed_ops_json", "[]") or "[]")
        results.append(item)
    return results


def replace_value_catalog(
    workspace_id: str,
    collection: str,
    values_by_column: dict[str, list[dict]],
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "DELETE FROM value_catalog WHERE workspace_id=? AND collection=?",
                (workspace_id, collection),
            )
            for column_name, values in values_by_column.items():
                for item in values:
                    conn.execute(
                        """
                        INSERT INTO value_catalog (
                            workspace_id, collection, column_name, normalized_value, raw_value, frequency
                        ) VALUES (?,?,?,?,?,?)
                        """,
                        (
                            workspace_id,
                            collection,
                            column_name,
                            item.get("normalized_value", ""),
                            item.get("raw_value", ""),
                            int(item.get("frequency", 0) or 0),
                        ),
                    )


def log_query_plan(
    workspace_id: str,
    collection: str,
    question: str,
    planner_source: str,
    plan: dict,
    validated: bool,
    validation_errors: list[str],
    sql_generated: str = "",
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO query_plans_log (
                    workspace_id, collection, question, planner_source, plan_json,
                    validated, validation_errors_json, sql_generated
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    workspace_id,
                    collection,
                    question,
                    planner_source,
                    json.dumps(plan, ensure_ascii=False),
                    1 if validated else 0,
                    json.dumps(validation_errors, ensure_ascii=False),
                    sql_generated,
                ),
            )


def list_collection_stats(workspace_id: str) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT
                collection,
                COUNT(*) AS documents,
                SUM(chunks_indexed) AS chunks_indexed,
                MAX(updated_at) AS updated_at
            FROM documents
            WHERE workspace_id=? AND status='indexed'
            GROUP BY collection
            ORDER BY collection
            """,
            (workspace_id,),
        ).fetchall()
    return [
        {
            "collection": row["collection"],
            "documents": row["documents"],
            "chunks_indexed": row["chunks_indexed"] or 0,
            "updated_at": row["updated_at"] or "",
        }
        for row in rows
    ]


def list_collections_with_models(workspace_id: str) -> list[dict]:
    """Return distinct (collection, embedding_model) pairs for indexed documents."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT collection, embedding_model
            FROM documents
            WHERE workspace_id=? AND status='indexed'
            ORDER BY collection
            """,
            (workspace_id,),
        ).fetchall()
    return [
        {"collection": row["collection"], "embedding_model": row["embedding_model"]}
        for row in rows
    ]


def create_ingestion_job(
    *,
    workspace_id: str,
    collection: str,
    doc_id: str,
    filename: str,
    embedding_model: str,
) -> IngestionJob:
    job_id = str(uuid.uuid4())
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    id, workspace_id, collection, doc_id, filename, embedding_model, status
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (job_id, workspace_id, collection, doc_id, filename, embedding_model, "queued"),
            )
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()
    assert row is not None
    return _row_to_job(row)


def update_ingestion_job(
    job_id: str,
    *,
    status: str,
    chunks_indexed: int | None = None,
    error: str | None = None,
    started: bool = False,
    finished: bool = False,
) -> IngestionJob | None:
    assignments = ["status=?", "error=?"]
    params: list[object] = [status, error or ""]
    if chunks_indexed is not None:
        assignments.append("chunks_indexed=?")
        params.append(chunks_indexed)
    if started:
        assignments.append("started_at=datetime('now','localtime')")
    if finished:
        assignments.append("finished_at=datetime('now','localtime')")

    params.append(job_id)
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                f"UPDATE ingestion_jobs SET {', '.join(assignments)} WHERE id=?",
                params,
            )
        row = conn.execute("SELECT * FROM ingestion_jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_job(row) if row else None


def get_ingestion_job(job_id: str, workspace_id: str) -> IngestionJob | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE id=? AND workspace_id=?",
            (job_id, workspace_id),
        ).fetchone()
    return _row_to_job(row) if row else None


def list_ingestion_jobs(workspace_id: str, limit: int = 50) -> list[IngestionJob]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM ingestion_jobs WHERE workspace_id=? ORDER BY created_at DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def record_audit(
    *,
    workspace_id: str,
    actor: str,
    event_type: str,
    resource_type: str,
    resource_id: str,
    details: dict | None = None,
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO audit_events (
                    workspace_id, actor, event_type, resource_type, resource_id, details_json
                ) VALUES (?,?,?,?,?,?)
                """,
                (
                    workspace_id,
                    actor,
                    event_type,
                    resource_type,
                    resource_id,
                    json.dumps(details or {}, ensure_ascii=False),
                ),
            )


def list_audit_events(workspace_id: str, limit: int = 100) -> list[AuditEvent]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE workspace_id=? ORDER BY created_at DESC, id DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
    return [_row_to_audit(row) for row in rows]


# ── Document nodes & summaries ──────────────────────────────────────────────

def upsert_document_node(
    *,
    workspace_id: str,
    collection: str,
    doc_id: str,
    node_id: str,
    node_type: str,
    label: str = "",
    numeral: str = "",
    path: str = "",
    parent_node_id: str = "",
    text_length: int = 0,
    articles: list[str] | None = None,
    refs: list[str] | None = None,
    tree_json: str = "{}",
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO document_nodes (
                    id, workspace_id, collection, doc_id, node_type, label, numeral,
                    path, parent_node_id, text_length, articles_json, refs_json, tree_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(workspace_id, collection, doc_id, node_type, label)
                DO UPDATE SET
                    numeral=excluded.numeral,
                    path=excluded.path,
                    parent_node_id=excluded.parent_node_id,
                    text_length=excluded.text_length,
                    articles_json=excluded.articles_json,
                    refs_json=excluded.refs_json,
                    tree_json=excluded.tree_json
                """,
                (
                    node_id, workspace_id, collection, doc_id, node_type, label,
                    numeral, path, parent_node_id, text_length,
                    json.dumps(articles or [], ensure_ascii=False),
                    json.dumps(refs or [], ensure_ascii=False),
                    tree_json,
                ),
            )


def list_document_nodes(
    workspace_id: str,
    collection: str,
    doc_id: str | None = None,
    node_type: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM document_nodes WHERE workspace_id=? AND collection=?"
    params: list[str] = [workspace_id, collection]
    if doc_id:
        query += " AND doc_id=?"
        params.append(doc_id)
    if node_type:
        query += " AND node_type=?"
        params.append(node_type)
    query += " ORDER BY created_at"
    with closing(_connect()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "collection": row["collection"],
            "doc_id": row["doc_id"],
            "node_type": row["node_type"],
            "label": row["label"],
            "numeral": row["numeral"],
            "path": row["path"],
            "parent_node_id": row["parent_node_id"],
            "text_length": row["text_length"],
            "articles": json.loads(row["articles_json"]),
            "refs": json.loads(row["refs_json"]),
            "tree_json": row["tree_json"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_document_node(workspace_id: str, collection: str, node_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM document_nodes WHERE workspace_id=? AND collection=? AND id=?",
            (workspace_id, collection, node_id),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "collection": row["collection"],
        "doc_id": row["doc_id"],
        "node_type": row["node_type"],
        "label": row["label"],
        "numeral": row["numeral"],
        "path": row["path"],
        "parent_node_id": row["parent_node_id"],
        "text_length": row["text_length"],
        "articles": json.loads(row["articles_json"]),
        "refs": json.loads(row["refs_json"]),
        "tree_json": row["tree_json"],
        "created_at": row["created_at"],
    }


def delete_document_nodes(workspace_id: str, collection: str, doc_id: str) -> int:
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                "DELETE FROM document_nodes WHERE workspace_id=? AND collection=? AND doc_id=?",
                (workspace_id, collection, doc_id),
            )
            return cursor.rowcount


def upsert_document_summary(
    *,
    workspace_id: str,
    collection: str,
    doc_id: str,
    node_id: str,
    node_type: str,
    label: str = "",
    path: str = "",
    resumo_executivo: str = "",
    resumo_juridico: str = "",
    pontos_chave: list[str] | None = None,
    artigos_cobertos: list[str] | None = None,
    obrigacoes: list[str] | None = None,
    restricoes: list[str] | None = None,
    definicoes: list[str] | None = None,
    text_length: int = 0,
    source_hash: str = "",
    source_text_length: int = 0,
    status: str = "generated",
    invalid_reason: str = "",
    generation_meta: dict | None = None,
) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO document_summaries (
                    workspace_id, collection, doc_id, node_id, node_type, label, path,
                    resumo_executivo, resumo_juridico, pontos_chave_json, artigos_cobertos_json,
                    obrigacoes_json, restricoes_json, definicoes_json, text_length,
                    source_hash, source_text_length, status, invalid_reason, generation_meta_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(workspace_id, collection, doc_id, node_id)
                DO UPDATE SET
                    node_type=excluded.node_type,
                    label=excluded.label,
                    path=excluded.path,
                    resumo_executivo=excluded.resumo_executivo,
                    resumo_juridico=excluded.resumo_juridico,
                    pontos_chave_json=excluded.pontos_chave_json,
                    artigos_cobertos_json=excluded.artigos_cobertos_json,
                    obrigacoes_json=excluded.obrigacoes_json,
                    restricoes_json=excluded.restricoes_json,
                    definicoes_json=excluded.definicoes_json,
                    text_length=excluded.text_length,
                    source_hash=excluded.source_hash,
                    source_text_length=excluded.source_text_length,
                    status=excluded.status,
                    invalid_reason=excluded.invalid_reason,
                    generation_meta_json=excluded.generation_meta_json
                """,
                (
                    workspace_id, collection, doc_id, node_id, node_type, label, path,
                    resumo_executivo, resumo_juridico,
                    json.dumps(pontos_chave or [], ensure_ascii=False),
                    json.dumps(artigos_cobertos or [], ensure_ascii=False),
                    json.dumps(obrigacoes or [], ensure_ascii=False),
                    json.dumps(restricoes or [], ensure_ascii=False),
                    json.dumps(definicoes or [], ensure_ascii=False),
                    text_length,
                    source_hash,
                    source_text_length,
                    status,
                    invalid_reason,
                    json.dumps(generation_meta or {}, ensure_ascii=False),
                ),
            )


def list_document_summaries(
    workspace_id: str,
    collection: str,
    doc_id: str | None = None,
    node_type: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM document_summaries WHERE workspace_id=? AND collection=?"
    params: list[str] = [workspace_id, collection]
    if doc_id:
        query += " AND doc_id=?"
        params.append(doc_id)
    if node_type:
        query += " AND node_type=?"
        params.append(node_type)
    query += " ORDER BY created_at"
    with closing(_connect()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "collection": row["collection"],
            "doc_id": row["doc_id"],
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "label": row["label"],
            "path": row["path"],
            "resumo_executivo": row["resumo_executivo"],
            "resumo_juridico": row["resumo_juridico"],
            "pontos_chave": json.loads(row["pontos_chave_json"]),
            "artigos_cobertos": json.loads(row["artigos_cobertos_json"]),
            "obrigacoes": json.loads(row["obrigacoes_json"]),
            "restricoes": json.loads(row["restricoes_json"]),
            "definicoes": json.loads(row["definicoes_json"]),
            "text_length": row["text_length"],
            "source_hash": row["source_hash"] if "source_hash" in row.keys() else "",
            "source_text_length": row["source_text_length"] if "source_text_length" in row.keys() else row["text_length"],
            "status": row["status"] if "status" in row.keys() else "generated",
            "invalid_reason": row["invalid_reason"] if "invalid_reason" in row.keys() else "",
            "generation_meta": json.loads(row["generation_meta_json"]) if "generation_meta_json" in row.keys() else {},
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_summary_by_node(workspace_id: str, collection: str, node_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM document_summaries WHERE workspace_id=? AND collection=? AND node_id=?",
            (workspace_id, collection, node_id),
        ).fetchone()
    if not row:
        return None
    return {
        "node_id": row["node_id"],
        "node_type": row["node_type"],
        "label": row["label"],
        "path": row["path"],
        "resumo_executivo": row["resumo_executivo"],
        "resumo_juridico": row["resumo_juridico"],
        "pontos_chave": json.loads(row["pontos_chave_json"]),
        "artigos_cobertos": json.loads(row["artigos_cobertos_json"]),
        "obrigacoes": json.loads(row["obrigacoes_json"]),
        "restricoes": json.loads(row["restricoes_json"]),
        "definicoes": json.loads(row["definicoes_json"]),
        "text_length": row["text_length"],
        "source_hash": row["source_hash"] if "source_hash" in row.keys() else "",
        "source_text_length": row["source_text_length"] if "source_text_length" in row.keys() else row["text_length"],
        "status": row["status"] if "status" in row.keys() else "generated",
        "invalid_reason": row["invalid_reason"] if "invalid_reason" in row.keys() else "",
        "generation_meta": json.loads(row["generation_meta_json"]) if "generation_meta_json" in row.keys() else {},
    }


def find_summaries_by_label(
    workspace_id: str,
    collection: str,
    label_query: str,
    node_type: str | None = None,
) -> list[dict]:
    """Find summaries whose label contains the query string (case-insensitive)."""
    query = "SELECT * FROM document_summaries WHERE workspace_id=? AND collection=? AND label LIKE ?"
    params: list[str] = [workspace_id, collection, f"%{label_query}%"]
    if node_type:
        query += " AND node_type=?"
        params.append(node_type)
    query += " ORDER BY created_at"
    with closing(_connect()) as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "node_id": row["node_id"],
            "node_type": row["node_type"],
            "label": row["label"],
            "path": row["path"],
            "resumo_executivo": row["resumo_executivo"],
            "resumo_juridico": row["resumo_juridico"],
            "pontos_chave": json.loads(row["pontos_chave_json"]),
            "artigos_cobertos": json.loads(row["artigos_cobertos_json"]),
            "obrigacoes": json.loads(row["obrigacoes_json"]),
            "restricoes": json.loads(row["restricoes_json"]),
            "definicoes": json.loads(row["definicoes_json"]),
            "text_length": row["text_length"],
            "source_hash": row["source_hash"] if "source_hash" in row.keys() else "",
            "source_text_length": row["source_text_length"] if "source_text_length" in row.keys() else row["text_length"],
            "status": row["status"] if "status" in row.keys() else "generated",
            "invalid_reason": row["invalid_reason"] if "invalid_reason" in row.keys() else "",
            "generation_meta": json.loads(row["generation_meta_json"]) if "generation_meta_json" in row.keys() else {},
        }
        for row in rows
    ]


def delete_document_summaries(workspace_id: str, collection: str, doc_id: str) -> int:
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                "DELETE FROM document_summaries WHERE workspace_id=? AND collection=? AND doc_id=?",
                (workspace_id, collection, doc_id),
            )
            return cursor.rowcount


init_db()
