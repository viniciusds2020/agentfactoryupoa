"""Structured store for tabular records (DuckDB)."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from src.config import get_settings
from src.utils import get_logger, log_event

logger = get_logger(__name__)

_CONN = None


def _normalize_identifier(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    if not normalized:
        return "coluna"
    if normalized[0].isdigit():
        return f"c_{normalized}"
    return normalized


def _table_name(collection: str) -> str:
    return f"{_normalize_identifier(collection)}_records"


def get_connection():
    global _CONN
    if _CONN is None:
        import duckdb

        db_path = Path(get_settings().structured_store_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _CONN = duckdb.connect(str(db_path))
        _CONN.execute("PRAGMA threads=4")
    return _CONN


def ensure_table(collection: str, column_names: list[str]) -> None:
    conn = get_connection()
    table = _table_name(collection)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            doc_id TEXT,
            row_index INTEGER,
            page_number INTEGER,
            raw_row TEXT,
            texto_canonico TEXT
        )
        """
    )

    existing = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
    for col in column_names:
        norm = _normalize_identifier(col)
        if norm not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {norm} TEXT")


def upsert_records(collection: str, doc_id: str, records: list, columns: list[str]) -> int:
    if not records:
        return 0
    try:
        ensure_table(collection, columns)
        delete_by_doc_id(collection, doc_id)
        conn = get_connection()
    except Exception:
        return 0
    table = _table_name(collection)
    norm_cols = [_normalize_identifier(c) for c in columns]

    insert_cols = ["doc_id", "row_index", "page_number", "raw_row", "texto_canonico", *norm_cols]
    placeholders = ", ".join(["?"] * len(insert_cols))
    sql = f"INSERT INTO {table} ({', '.join(insert_cols)}) VALUES ({placeholders})"

    rows_to_insert = []
    for rec in records:
        fields = getattr(rec, "fields", {}) if not isinstance(rec, dict) else rec.get("fields", {})
        row = [
            doc_id,
            getattr(rec, "row_index", None) if not isinstance(rec, dict) else rec.get("row_index"),
            getattr(rec, "page_number", None) if not isinstance(rec, dict) else rec.get("page_number"),
            getattr(rec, "raw_row", "") if not isinstance(rec, dict) else rec.get("raw_row", ""),
            getattr(rec, "texto_canonico", "") if not isinstance(rec, dict) else rec.get("texto_canonico", ""),
        ]
        for col in norm_cols:
            row.append(fields.get(col, "") or fields.get(col.lower(), ""))
        rows_to_insert.append(row)

    conn.executemany(sql, rows_to_insert)
    log_event(logger, 20, "Structured records upserted", collection=collection, doc_id=doc_id, rows=len(rows_to_insert))
    return len(rows_to_insert)


def query_structured(collection: str, filters: dict, limit: int = 20) -> list[dict]:
    table = _table_name(collection)
    try:
        conn = get_connection()
        conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
    except Exception:
        return []

    conditions = []
    params: list[object] = []
    for key, value in (filters or {}).items():
        col = _normalize_identifier(key)
        conditions.append(f"LOWER(CAST({col} AS VARCHAR)) LIKE ?")
        params.append(f"%{str(value).strip().lower()}%")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT * FROM {table} {where_clause} LIMIT ?",
        [*params, max(1, limit)],
    ).fetchall()
    columns = [d[0] for d in conn.description]
    return [dict(zip(columns, row)) for row in rows]


def delete_by_doc_id(collection: str, doc_id: str) -> int:
    table = _table_name(collection)
    try:
        conn = get_connection()
        cursor = conn.execute(f"DELETE FROM {table} WHERE doc_id = ?", [doc_id])
        return cursor.rowcount
    except Exception:
        return 0


def has_structured_data(collection: str) -> bool:
    table = _table_name(collection)
    try:
        conn = get_connection()
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return bool(row and row[0] > 0)
    except Exception:
        return False
