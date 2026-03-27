"""SQLite-backed conversation history.

Schema
------
conversations : id, title, collection, embedding_model, created_at, updated_at
messages      : id, conversation_id, role, content, sources_json, created_at

DB path: data/history.db  (auto-created on first use)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.utils import get_logger, log_event

_DB_PATH = Path("data/history.db")
logger = get_logger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with closing(_connect()) as conn:
        with conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id             TEXT PRIMARY KEY,
                    workspace_id   TEXT NOT NULL DEFAULT 'default',
                    title          TEXT NOT NULL,
                    collection     TEXT NOT NULL DEFAULT '',
                    embedding_model TEXT NOT NULL DEFAULT '',
                    created_at     TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    updated_at     TEXT NOT NULL DEFAULT (datetime('now','localtime'))
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role            TEXT NOT NULL CHECK (role IN ('user','assistant')),
                    content         TEXT NOT NULL,
                    sources_json    TEXT NOT NULL DEFAULT '[]',
                    created_at      TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id);
            """)
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)").fetchall()}
            if "workspace_id" not in cols:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN workspace_id TEXT NOT NULL DEFAULT 'default'"
                )


# ── Domain types ──────────────────────────────────────────────────────────────

@dataclass
class Conversation:
    id: str
    workspace_id: str
    title: str
    collection: str
    embedding_model: str
    created_at: str
    updated_at: str


@dataclass
class StoredMessage:
    role: str
    content: str
    sources: list[dict]  # plain dicts — serialised from Source dataclass


# ── Write operations ──────────────────────────────────────────────────────────

def create_conversation(
    workspace_id: str,
    collection: str,
    embedding_model: str,
    title: str = "",
) -> str:
    conv_id = str(uuid.uuid4())
    if not title:
        title = f"Conversa {datetime.now().strftime('%d/%m %H:%M')}"
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "INSERT INTO conversations (id, workspace_id, title, collection, embedding_model) VALUES (?,?,?,?,?)",
                (conv_id, workspace_id, title, collection, embedding_model),
            )
    log_event(
        logger,
        20,
        "Conversation created",
        conversation_id=conv_id,
        workspace_id=workspace_id,
        collection=collection,
        embedding_model=embedding_model,
    )
    return conv_id


def rename_conversation(conv_id: str, title: str) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "UPDATE conversations SET title=?, updated_at=datetime('now','localtime') WHERE id=?",
                (title, conv_id),
            )
    log_event(logger, 20, "Conversation renamed", conversation_id=conv_id, title=title)


def save_message(conv_id: str, role: str, content: str, sources: list | None = None) -> None:
    sources = sources or []
    sources_json = json.dumps([_src_to_dict(s) for s in sources], ensure_ascii=False)
    with closing(_connect()) as conn:
        with conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources_json) VALUES (?,?,?,?)",
                (conv_id, role, content, sources_json),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=datetime('now','localtime') WHERE id=?",
                (conv_id,),
            )
    log_event(
        logger,
        20,
        "Message saved",
        conversation_id=conv_id,
        role=role,
        content_length=len(content),
        sources=len(sources),
    )


def delete_conversation(conv_id: str) -> None:
    with closing(_connect()) as conn:
        with conn:
            conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    log_event(logger, 20, "Conversation deleted", conversation_id=conv_id)


def purge_old_conversations(days: int = 90) -> int:
    """Delete conversations not updated in the last *days* days. Returns count deleted."""
    with closing(_connect()) as conn:
        with conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE updated_at < datetime('now', 'localtime', '-' || ? || ' days')",
                (days,),
            )
            deleted = cursor.rowcount
    if deleted:
        log_event(logger, 20, "Old conversations purged", days=days, deleted=deleted)
    return deleted


# ── Read operations ───────────────────────────────────────────────────────────

def search_conversations(workspace_id: str, query: str, limit: int = 30) -> list[Conversation]:
    """Full-text search across conversation titles and message content."""
    pattern = f"%{query.strip()}%"
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT c.*
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.workspace_id = ?
              AND (
                   c.title LIKE ?
                OR c.collection LIKE ?
                OR m.content LIKE ?
              )
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (workspace_id, pattern, pattern, pattern, limit),
        ).fetchall()
    return [
        Conversation(
            id=r["id"],
            workspace_id=r["workspace_id"],
            title=r["title"],
            collection=r["collection"],
            embedding_model=r["embedding_model"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


def list_conversations(workspace_id: str, limit: int = 40) -> list[Conversation]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE workspace_id=? ORDER BY updated_at DESC LIMIT ?",
            (workspace_id, limit),
        ).fetchall()
    return [
        Conversation(
            id=r["id"],
            workspace_id=r["workspace_id"],
            title=r["title"],
            collection=r["collection"],
            embedding_model=r["embedding_model"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


def load_messages(conv_id: str) -> list[StoredMessage]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT role, content, sources_json FROM messages "
            "WHERE conversation_id=? ORDER BY id",
            (conv_id,),
        ).fetchall()
    return [
        StoredMessage(
            role=r["role"],
            content=r["content"],
            sources=json.loads(r["sources_json"]),
        )
        for r in rows
    ]


def get_conversation(conv_id: str, workspace_id: str | None = None) -> Conversation | None:
    with closing(_connect()) as conn:
        if workspace_id is None:
            row = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id=? AND workspace_id=?",
                (conv_id, workspace_id),
            ).fetchone()
    if not row:
        return None
    return Conversation(
        id=row["id"],
        workspace_id=row["workspace_id"],
        title=row["title"],
        collection=row["collection"],
        embedding_model=row["embedding_model"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _src_to_dict(src) -> dict:
    if isinstance(src, dict):
        return src
    return {
        "chunk_id": getattr(src, "chunk_id", ""),
        "doc_id":   getattr(src, "doc_id", ""),
        "excerpt":  getattr(src, "excerpt", ""),
        "score":    getattr(src, "score", 0.0),
        "metadata": getattr(src, "metadata", {}),
    }


# Auto-initialise on import
init_db()
