"""Structured store for tabular records (DuckDB)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import get_settings
from src.observability import metrics
from src.table_planner import build_query_plan
from src.table_sql_builder import build_sql_for_plan
from src.table_semantics import infer_semantic_column, infer_table_type, semantic_role_for_type
from src.table_validator import validate_query_plan
from src.utils import get_logger, log_event

logger = get_logger(__name__)

_CONN = None
_BACKEND = ""
_SYSTEM_COLUMNS = {"doc_id", "row_index", "page_number", "raw_row", "texto_canonico"}
_NUMERIC_HINTS = {
    "valor", "price", "amount", "total", "receita", "faturamento",
    "custo", "salario", "renda", "score", "nota",
}
_STATE_NAME_TO_UF = {
    "acre": "AC",
    "alagoas": "AL",
    "amapa": "AP",
    "amazonas": "AM",
    "bahia": "BA",
    "ceara": "CE",
    "distrito federal": "DF",
    "espirito santo": "ES",
    "goias": "GO",
    "maranhao": "MA",
    "mato grosso": "MT",
    "mato grosso do sul": "MS",
    "minas gerais": "MG",
    "para": "PA",
    "paraiba": "PB",
    "parana": "PR",
    "pernambuco": "PE",
    "piaui": "PI",
    "rio de janeiro": "RJ",
    "rio grande do norte": "RN",
    "rio grande do sul": "RS",
    "rondonia": "RO",
    "roraima": "RR",
    "santa catarina": "SC",
    "sao paulo": "SP",
    "sergipe": "SE",
    "tocantins": "TO",
}


@dataclass
class ColumnProfile:
    name: str
    data_type: Literal["numeric", "text", "date", "unknown"]
    role: Literal["metric", "dimension", "identifier", "date", "text"]
    semantic_type: str
    unit: str
    aliases: list[str]
    examples: list[str]
    description: str


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
    global _CONN, _BACKEND
    if _CONN is None:
        db_path = Path(get_settings().structured_store_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import duckdb

            _CONN = duckdb.connect(str(db_path))
            _CONN.execute("PRAGMA threads=4")
            _BACKEND = "duckdb"
        except Exception:
            import sqlite3

            _CONN = sqlite3.connect(str(db_path))
            _BACKEND = "sqlite"
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

    available_columns = set(_list_user_columns(collection))
    profiles = _infer_column_profiles(collection)
    conditions = []
    params: list[object] = []
    for key, value in (filters or {}).items():
        col = _resolve_filter_column(key, available_columns, profiles)
        if not col:
            continue
        conditions.append(f"LOWER(CAST({col} AS VARCHAR)) LIKE ?")
        params.append(f"%{str(value).strip().lower()}%")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    cursor = conn.execute(
        f"SELECT * FROM {table} {where_clause} LIMIT ?",
        [*params, max(1, limit)],
    )
    rows = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def _resolve_filter_column(key: str, available_columns: set[str], profiles: list[ColumnProfile]) -> str | None:
    normalized = _normalize_identifier(key)
    if normalized in available_columns:
        return normalized

    semantic_aliases: dict[str, str] = {
        "codigo": "identifier",
        "code": "identifier",
        "id": "identifier",
        "cobertura": "coverage_rule",
        "autorizacao": "authorization_rule",
        "prazo": "deadline_rule",
        "descricao": "catalog_title",
        "titulo": "catalog_title",
    }
    semantic_target = semantic_aliases.get(normalized)
    if semantic_target:
        match = next((profile.name for profile in profiles if profile.semantic_type == semantic_target and profile.name in available_columns), None)
        if match:
            return match

    profile_match = next(
        (
            profile.name
            for profile in profiles
            if profile.name in available_columns and (
                normalized in profile.aliases
                or any(alias == normalized or normalized in alias or alias in normalized for alias in profile.aliases)
            )
        ),
        None,
    )
    if profile_match:
        return profile_match
    return None


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


def _normalize_text(value: str) -> str:
    return _normalize_identifier(value).replace("_", " ").strip()


def _list_user_columns(collection: str) -> list[str]:
    table = _table_name(collection)
    try:
        conn = get_connection()
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:
        return []
    return [row[1] for row in rows if row[1] not in _SYSTEM_COLUMNS]


def _sample_rows(collection: str, limit: int = 200) -> list[dict]:
    table = _table_name(collection)
    try:
        conn = get_connection()
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT ?", [max(1, limit)])
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
    except Exception:
        return []
    return [dict(zip(columns, row)) for row in rows]


def _is_numeric_value(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    text = text.replace(".", "").replace(",", ".") if re.fullmatch(r"\d{1,3}(?:\.\d{3})*,\d+", text) else text
    return bool(re.fullmatch(r"-?\d+(?:[.,]\d+)?", text))


def _as_float_sql(column: str) -> str:
    if _BACKEND == "sqlite":
        text_expr = f"TRIM(CAST({column} AS TEXT))"
        return (
            "CASE "
            f"WHEN INSTR({text_expr}, ',') > 0 AND INSTR({text_expr}, '.') > 0 "
            f"THEN CAST(REPLACE(REPLACE({text_expr}, '.', ''), ',', '.') AS REAL) "
            f"WHEN INSTR({text_expr}, ',') > 0 "
            f"THEN CAST(REPLACE({text_expr}, ',', '.') AS REAL) "
            f"ELSE CAST({text_expr} AS REAL) END"
        )
    text_expr = f"TRIM(CAST({column} AS VARCHAR))"
    return (
        "CASE "
        f"WHEN regexp_matches({text_expr}, '^-?[0-9]{{1,3}}(?:\\.[0-9]{{3}})+(?:,[0-9]+)?$') "
        f"THEN TRY_CAST(REPLACE(REPLACE({text_expr}, '.', ''), ',', '.') AS DOUBLE) "
        f"WHEN regexp_matches({text_expr}, '^-?[0-9]+,[0-9]+$') "
        f"THEN TRY_CAST(REPLACE({text_expr}, ',', '.') AS DOUBLE) "
        f"ELSE TRY_CAST({text_expr} AS DOUBLE) END"
    )


def _infer_column_profiles(collection: str) -> list[ColumnProfile]:
    columns = _list_user_columns(collection)
    rows = _sample_rows(collection)
    profiles: list[ColumnProfile] = []
    for column in columns:
        values = [str(row.get(column, "") or "").strip() for row in rows]
        non_empty = [value for value in values if value]
        numeric_ratio = (
            sum(1 for value in non_empty if _is_numeric_value(value)) / len(non_empty)
            if non_empty else 0.0
        )
        normalized_name = _normalize_identifier(column)
        examples = list(dict.fromkeys(non_empty[:5]))
        if re.search(r"(data|date)", normalized_name):
            data_type = "date"
        elif re.search(r"(^id$|^id_|_id$|codigo|code)", normalized_name):
            data_type = "text"
        elif numeric_ratio >= 0.8:
            data_type = "numeric"
        elif re.search(r"(status|situacao|segmento|cidade|estado|uf|categoria|tipo)", normalized_name):
            data_type = "text"
        else:
            data_type = "text"
        semantic = infer_semantic_column(column, numeric_ratio)
        role: Literal["metric", "dimension", "identifier", "date", "text"] = semantic_role_for_type(semantic.semantic_type)
        aliases = list({
            normalized_name,
            _normalize_text(column),
            normalized_name.replace("_", ""),
            *normalized_name.split("_"),
        })
        profiles.append(
            ColumnProfile(
                column,
                data_type,
                role,
                semantic.semantic_type,
                semantic.unit,
                [a for a in aliases if a],
                examples,
                semantic.description,
            )
        )
    return profiles


def get_column_profiles(collection: str) -> list[dict]:
    return [
        {
            "name": profile.name,
            "display_name": profile.name,
            "data_type": profile.data_type,
            "role": profile.role,
            "semantic_type": profile.semantic_type,
            "unit": profile.unit,
            "aliases": profile.aliases,
            "examples": profile.examples,
            "description": profile.description,
            "allowed_operations": _allowed_operations(profile.semantic_type),
            "cardinality": _column_cardinality(collection, profile.name),
        }
        for profile in _infer_column_profiles(collection)
    ]


def get_table_profile(collection: str, context_hint: str = "") -> dict:
    profiles = get_column_profiles(collection)
    return {
        "table_name": _table_name(collection),
        "table_type": infer_table_type(profiles, context_hint=context_hint),
    }


def _allowed_operations(semantic_type: str) -> list[str]:
    if semantic_type == "measure_currency":
        return ["sum", "avg", "min", "max"]
    if semantic_type in {"measure_age_years", "measure_score", "measure_number", "measure_percentage"}:
        return ["avg", "min", "max", "sum"]
    if semantic_type in {"catalog_title", "catalog_attribute", "coverage_rule", "authorization_rule", "deadline_rule"}:
        return ["lookup", "summary"]
    if semantic_type.startswith("dimension_") or semantic_type == "flag_boolean":
        return ["filter", "distinct", "group_by"]
    if semantic_type == "identifier":
        return ["filter", "distinct", "lookup"]
    if semantic_type in {"date", "datetime"}:
        return ["filter", "group_by", "distinct"]
    return ["filter"]


def _column_cardinality(collection: str, column: str) -> int:
    table = _table_name(collection)
    try:
        conn = get_connection()
        row = conn.execute(
            f"SELECT COUNT(DISTINCT CAST({column} AS VARCHAR)) FROM {table} WHERE CAST({column} AS VARCHAR) <> ''"
        ).fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def get_value_catalog(collection: str, limit_per_column: int = 25) -> dict[str, list[dict]]:
    table = _table_name(collection)
    catalog: dict[str, list[dict]] = {}
    for profile in _infer_column_profiles(collection):
        if profile.role not in {"dimension", "identifier", "text"}:
            continue
        try:
            conn = get_connection()
            rows = conn.execute(
                f"""
                SELECT CAST({profile.name} AS VARCHAR) AS raw_value, COUNT(*) AS frequency
                FROM {table}
                WHERE CAST({profile.name} AS VARCHAR) <> ''
                GROUP BY CAST({profile.name} AS VARCHAR)
                ORDER BY frequency DESC, raw_value ASC
                LIMIT ?
                """,
                [max(1, limit_per_column)],
            ).fetchall()
        except Exception:
            rows = []
        catalog[profile.name] = [
            {
                "normalized_value": _normalize_text(str(row[0])),
                "raw_value": str(row[0]),
                "frequency": int(row[1] or 0),
            }
            for row in rows
            if row and row[0] is not None
        ]
    return catalog


def persist_table_semantics(collection: str, workspace_id: str = "default", context_hint: str = "") -> None:
    try:
        from src import controlplane
        from src.table_semantics import infer_subject_label

        if not context_hint:
            context_hint = controlplane.get_collection_context(workspace_id, collection)
        profiles = get_column_profiles(collection)
        table_profile = get_table_profile(collection, context_hint=context_hint)
        controlplane.upsert_table_profile(
            workspace_id=workspace_id,
            collection=collection,
            table_name=table_profile["table_name"],
            base_context=context_hint,
            subject_label=infer_subject_label("", context_hint),
        )
        controlplane.replace_column_profiles(workspace_id, collection, profiles)
        controlplane.replace_value_catalog(workspace_id, collection, get_value_catalog(collection))
    except Exception as exc:
        log_event(
            logger,
            30,
            "Failed to persist table semantics",
            collection=collection,
            workspace_id=workspace_id,
            error=str(exc),
        )


def _find_profile(profiles: list[ColumnProfile], token: str) -> ColumnProfile | None:
    token_norm = _normalize_text(token)
    for profile in profiles:
        if token_norm in profile.aliases:
            return profile
    for profile in profiles:
        if any(alias in token_norm or token_norm in alias for alias in profile.aliases):
            return profile
    return None


def _resolve_metric_column(question: str, profiles: list[ColumnProfile], context_hint: str = "") -> ColumnProfile | None:
    q_norm = _normalize_text(f"{question} {context_hint}".strip())
    metric_profiles = [p for p in profiles if p.role == "metric"]
    for profile in metric_profiles:
        if any(alias and alias in q_norm for alias in profile.aliases):
            return profile
    hinted = [
        profile for profile in metric_profiles
        if any(hint in profile.name for hint in _NUMERIC_HINTS)
    ]
    if len(hinted) == 1:
        return hinted[0]
    if len(metric_profiles) == 1:
        return metric_profiles[0]
    return hinted[0] if hinted else None


def _resolve_dimension_column(question: str, profiles: list[ColumnProfile], keyword: str) -> ColumnProfile | None:
    q_norm = _normalize_text(question)
    if keyword not in q_norm:
        return None
    for profile in profiles:
        if profile.role in {"dimension", "text"} and any(alias == keyword or keyword in alias for alias in profile.aliases):
            return profile
    return None


def _resolve_inventory_column(question: str, profiles: list[ColumnProfile], context_hint: str = "") -> ColumnProfile | None:
    q_norm = _normalize_text(f"{question} {context_hint}".strip())
    if not re.search(r"\b(quais|quais sao|liste|listar|mostre|mostrar)\b", q_norm):
        return None
    explicit_aliases = [
        "estado",
        "estados",
        "uf",
        "ufs",
        "cidade",
        "cidades",
        "segmento",
        "segmentos",
        "status",
        "categoria",
        "categorias",
        "tipo",
        "tipos",
    ]
    for token in explicit_aliases:
        if not re.search(rf"\b{re.escape(token)}\b", q_norm):
            continue
        singular = token[:-1] if token.endswith("s") else token
        profile = _resolve_dimension_column(singular, profiles, singular) or _find_profile(profiles, singular)
        if profile:
            return profile
    preferred_dimensions = [p for p in profiles if p.role in {"dimension", "identifier", "text"}]
    for profile in preferred_dimensions:
        aliases = [alias for alias in profile.aliases if alias]
        for alias in aliases:
            plural = f"{alias}s" if not alias.endswith("s") else alias
            if re.search(rf"\b{re.escape(alias)}\b", q_norm) or re.search(rf"\b{re.escape(plural)}\b", q_norm):
                return profile
    return None


def _distinct_values(collection: str, column: str, limit: int = 200) -> list[str]:
    table = _table_name(collection)
    try:
        conn = get_connection()
        rows = conn.execute(
            f"SELECT DISTINCT CAST({column} AS VARCHAR) AS v FROM {table} WHERE {column} IS NOT NULL AND CAST({column} AS VARCHAR) <> '' LIMIT ?",
            [max(1, limit)],
        ).fetchall()
    except Exception:
        return []
    return [str(row[0]) for row in rows if row and row[0] is not None]


def _resolve_state_filter(question: str, profiles: list[ColumnProfile], collection: str) -> dict | None:
    q_norm = _normalize_text(question)
    mentions_state_name = any(state_name in q_norm for state_name in _STATE_NAME_TO_UF)
    mentions_uf = any(re.search(rf"\b{re.escape(uf.lower())}\b", q_norm) for uf in _STATE_NAME_TO_UF.values())
    if "estado" not in q_norm and " uf " not in f" {q_norm} " and not mentions_state_name and not mentions_uf:
        return None
    state_col = _find_profile(profiles, "estado") or _find_profile(profiles, "uf")
    if not state_col:
        return None
    distinct = _distinct_values(collection, state_col.name)
    distinct_norm = {_normalize_text(v): v for v in distinct}
    for state_name, uf in _STATE_NAME_TO_UF.items():
        if state_name in q_norm:
            if uf.lower() in {str(v).lower() for v in distinct}:
                return {"column": state_col.name, "operator": "=", "value": uf}
            for norm_val, raw in distinct_norm.items():
                if state_name == norm_val:
                    return {"column": state_col.name, "operator": "=", "value": raw}
            return {"column": state_col.name, "operator": "=", "value": uf}
    for uf in _STATE_NAME_TO_UF.values():
        if re.search(rf"\b{re.escape(uf.lower())}\b", q_norm):
            return {"column": state_col.name, "operator": "=", "value": uf}
    return None


def _resolve_comparative_filters(question: str, profiles: list[ColumnProfile]) -> list[dict]:
    q_norm = _normalize_text(question).replace("_", " ")
    filters: list[dict] = []

    age_profile = next((p for p in profiles if p.semantic_type == "measure_age_years"), None)
    if age_profile:
        patterns = [
            (r"\b(?:acima de|mais de|maior(?:es)? que|maior(?:es)? de)\s+(\d+(?:[.,]\d+)?)\s+anos?\b", ">"),
            (r"\b(?:abaixo de|menos de|menor(?:es)? que|menor(?:es)? de)\s+(\d+(?:[.,]\d+)?)\s+anos?\b", "<"),
        ]
        for pattern, operator in patterns:
            match = re.search(pattern, q_norm)
            if match:
                filters.append(
                    {
                        "column": age_profile.name,
                        "operator": operator,
                        "value": match.group(1).replace(",", "."),
                    }
                )
                break

    for profile in profiles:
        if profile.role != "metric":
            continue
        aliases = [alias.replace("_", " ") for alias in profile.aliases if alias]
        for alias in aliases:
            patterns = [
                (rf"\b{re.escape(alias)}\b.*?\b(?:acima de|mais de|maior que)\s+(\d+(?:[.,]\d+)?)\b", ">"),
                (rf"\b{re.escape(alias)}\b.*?\b(?:abaixo de|menos de|menor que)\s+(\d+(?:[.,]\d+)?)\b", "<"),
                (rf"\b(?:acima de|mais de|maior que)\s+(\d+(?:[.,]\d+)?)\b.*?\b{re.escape(alias)}\b", ">"),
                (rf"\b(?:abaixo de|menos de|menor que)\s+(\d+(?:[.,]\d+)?)\b.*?\b{re.escape(alias)}\b", "<"),
            ]
            matched = False
            for pattern, operator in patterns:
                match = re.search(pattern, q_norm)
                if match:
                    filters.append(
                        {
                            "column": profile.name,
                            "operator": operator,
                            "value": match.group(1).replace(",", "."),
                        }
                    )
                    matched = True
                    break
            if matched:
                break

    date_profile = next((p for p in profiles if p.semantic_type in {"date", "datetime"}), None)
    if date_profile:
        between_match = re.search(r"\bentre\s+(\d{4})\s+e\s+(\d{4})\b", q_norm)
        if between_match:
            filters.append(
                {
                    "column": date_profile.name,
                    "operator": "between",
                    "value": f"{between_match.group(1)}-01-01|{between_match.group(2)}-12-31",
                }
            )
        else:
            after_match = re.search(r"\b(?:apos|a partir de|depois de)\s+(\d{4}(?:-\d{2}(?:-\d{2})?)?)\b", q_norm)
            before_match = re.search(r"\b(?:antes de|ate|at[eé])\s+(\d{4}(?:-\d{2}(?:-\d{2})?)?)\b", q_norm)
            if after_match:
                filters.append({"column": date_profile.name, "operator": ">=", "value": after_match.group(1)})
            if before_match:
                filters.append({"column": date_profile.name, "operator": "<=", "value": before_match.group(1)})

    unique: dict[str, dict] = {}
    for flt in filters:
        key = f"{flt['column']}::{flt.get('operator', '=')}"
        unique[key] = flt
    return list(unique.values())


def _resolve_generic_filters(question: str, profiles: list[ColumnProfile], collection: str) -> list[dict]:
    filters: list[dict] = []
    q_norm = _normalize_text(question)
    state_filter = _resolve_state_filter(question, profiles, collection)
    explicit_state_names = [name for name in _STATE_NAME_TO_UF if f"estado do {name}" in q_norm or f"estado de {name}" in q_norm]
    if state_filter:
        filters.append(state_filter)
    for profile in profiles:
        if profile.name == (state_filter or {}).get("column"):
            continue
        if profile.role not in {"dimension", "text"}:
            continue
        distinct = _distinct_values(collection, profile.name, limit=100)
        for raw in distinct:
            norm_val = _normalize_text(raw)
            if explicit_state_names and norm_val in explicit_state_names:
                continue
            if state_filter and profile.semantic_type == "dimension_geo_city" and norm_val in _STATE_NAME_TO_UF:
                continue
            if not norm_val:
                continue
            if len(norm_val) <= 3:
                matched = bool(re.search(rf"\b{re.escape(norm_val)}\b", q_norm))
            else:
                matched = norm_val in q_norm
            if matched:
                filters.append({"column": profile.name, "operator": "=", "value": raw})
                break
    filters.extend(_resolve_comparative_filters(question, profiles))
    unique: dict[str, dict] = {}
    for flt in filters:
        unique[flt["column"]] = flt
    return list(unique.values())


def plan_query(collection: str, question: str, context_hint: str = "") -> dict | None:
    profiles = _infer_column_profiles(collection)
    if not profiles:
        return None

    profile_dicts = get_column_profiles(collection)
    filters = _resolve_generic_filters(question, profiles, collection)
    plan_model = build_query_plan(question, profile_dicts, filters, context_hint=context_hint)
    if not plan_model:
        return None

    validation = validate_query_plan(plan_model.model_dump(), profile_dicts)
    operation_by_intent = {
        "tabular_aggregate": "aggregate",
        "tabular_count": "aggregate",
        "tabular_groupby": "groupby",
        "tabular_rank": "rank",
        "tabular_distinct": "distinct",
        "tabular_schema": "schema",
        "tabular_describe_column": "describe_column",
        "tabular_compare": "compare",
        "catalog_lookup_by_id": "catalog_lookup",
        "catalog_lookup_by_title": "catalog_lookup",
        "catalog_field_lookup": "catalog_field_lookup",
        "catalog_record_summary": "catalog_record_summary",
        "catalog_compare": "catalog_compare",
    }
    normalized = dict(validation.normalized_plan)
    normalized["operation"] = operation_by_intent.get(normalized.get("intent", ""), "")
    normalized["planner_source"] = str(normalized.get("planner_source") or plan_model.planner_source)
    normalized["validated"] = validation.valid
    normalized["validation_errors"] = list(validation.errors)
    normalized["assumption"] = "; ".join(normalized.get("assumptions", []) or [])
    if normalized.get("intent") == "tabular_compare":
        group_by = set(normalized.get("group_by", []) or [])
        normalized["filters"] = [
            flt for flt in normalized.get("filters", []) or []
            if flt.get("column") not in group_by
        ]
    normalized["table_type"] = infer_table_type(profile_dicts, context_hint=context_hint)

    if validation.valid:
        metrics.increment("tabular_plan_success_rate")
    else:
        metrics.increment("tabular_validation_fail_rate")

    log_event(
        logger,
        20 if validation.valid else 30,
        "Tabular query plan built",
        collection=collection,
        planner_source=normalized.get("planner_source", ""),
        intent=normalized.get("intent", ""),
        operation=normalized.get("operation", ""),
        validated=validation.valid,
        errors=validation.errors,
    )
    if not validation.valid:
        return None
    return normalized


def execute_plan(collection: str, plan: dict) -> dict | None:
    table = _table_name(collection)
    columns = set(_list_user_columns(collection))
    if not columns:
        return None

    conn = get_connection()
    op = plan.get("operation")

    if op == "schema":
        profiles = get_column_profiles(collection)
        return {
            "operation": op,
            "columns": [item["name"] for item in profiles],
            "profiles": profiles,
            "count": len(profiles),
            "plan": plan,
            "sql_generated": "",
        }

    if op == "describe_column":
        profiles = get_column_profiles(collection)
        target = str(plan.get("target_column") or "").strip()
        matches = [item for item in profiles if item["name"] == target]
        return {
            "operation": op,
            "profiles": profiles,
            "target_column": target,
            "target_profile": matches[0] if matches else None,
            "plan": plan,
            "sql_generated": "",
        }

    if op in {"catalog_lookup", "catalog_field_lookup", "catalog_record_summary", "catalog_compare"}:
        profiles = get_column_profiles(collection)
        rows = _execute_catalog_plan(collection, plan, profiles)
        if not rows:
            return None
        if op == "catalog_compare":
            return {"operation": op, "rows": rows, "profiles": profiles, "plan": plan, "sql_generated": ""}
        record = rows[0]
        return {
            "operation": op,
            "record": record,
            "rows": rows,
            "profiles": profiles,
            "target_column": plan.get("target_column"),
            "plan": plan,
            "sql_generated": "",
        }

    sql, params = build_sql_for_plan(table, plan, backend=_BACKEND or "duckdb")
    if not sql:
        return None

    log_event(
        logger,
        20,
        "Tabular plan executed",
        collection=collection,
        operation=op,
        sql=sql,
    )

    if op == "distinct":
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        values = [str(row[0]) for row in rows if row and row[0] is not None]
        return {"operation": op, "values": values, "count": len(values), "plan": plan, "sql_generated": sql}

    if op == "aggregate":
        row = conn.execute(sql, params).fetchone()
        return {"operation": op, "value": row[0] if row else None, "plan": plan, "sql_generated": sql}

    if op in {"groupby", "rank", "compare"}:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        columns_out = [d[0] for d in cursor.description]
        return {"operation": op, "rows": [dict(zip(columns_out, row)) for row in rows], "plan": plan, "sql_generated": sql}

    return None


def _all_rows(collection: str, limit: int | None = None) -> list[dict]:
    table = _table_name(collection)
    try:
        conn = get_connection()
        sql = f"SELECT * FROM {table}"
        params: list[object] = []
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        columns = [d[0] for d in cursor.description]
    except Exception:
        return []
    return [dict(zip(columns, row)) for row in rows]


def _matches_filter(row: dict, flt: dict) -> bool:
    column = str(flt.get("column") or "")
    operator = str(flt.get("operator") or "=").strip()
    value = str(flt.get("value") or "")
    raw = row.get(column)
    text = _normalize_text(str(raw or ""))
    target = _normalize_text(value)
    if operator == "=":
        return text == target
    if operator == "in":
        values = {_normalize_text(item) for item in value.split("|") if item.strip()}
        return text in values
    return False


def _execute_catalog_plan(collection: str, plan: dict, profiles: list[dict]) -> list[dict]:
    rows = _all_rows(collection)
    if not rows:
        return []
    matched = rows
    for flt in plan.get("filters", []) or []:
        matched = [row for row in matched if _matches_filter(row, flt)]

    if matched:
        return matched[:10]

    # fallback for title lookup with normalized contains
    title_col = next((p["name"] for p in profiles if p.get("semantic_type") == "catalog_title"), "")
    if title_col:
        for flt in plan.get("filters", []) or []:
            if flt.get("column") != title_col:
                continue
            target = _normalize_text(str(flt.get("value") or ""))
            fuzzy = [row for row in rows if target and target in _normalize_text(str(row.get(title_col) or ""))]
            if fuzzy:
                return fuzzy[:10]
    return []
