"""Safe SQL builder for table-first plans."""
from __future__ import annotations


def _as_date_sql(column: str, backend: str) -> str:
    if backend == "sqlite":
        return f"date(CAST({column} AS TEXT))"
    return f"TRY_CAST({column} AS DATE)"


def _date_bucket_sql(column: str, backend: str, grain: str | None) -> str:
    date_expr = _as_date_sql(column, backend)
    if grain == "year":
        return f"strftime({date_expr}, '%Y')"
    if grain == "month":
        return f"strftime({date_expr}, '%Y-%m')"
    return f"strftime({date_expr}, '%Y-%m-%d')"


def _as_float_sql(column: str, backend: str) -> str:
    if backend == "sqlite":
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


def build_sql_for_plan(table: str, plan: dict, backend: str = "duckdb") -> tuple[str, list[object]]:
    filters = plan.get("filters", []) or []
    where_parts: list[str] = []
    params: list[object] = []
    for flt in filters:
        operator = str(flt.get("operator", "=")).strip()
        column = str(flt["column"])
        value = flt.get("value", "")
        if operator in {">", "<", ">=", "<="}:
            if column in {"prazo_dias"}:
                where_parts.append(f"CAST({column} AS INTEGER) {operator} ?")
                params.append(int(float(str(value).replace(",", "."))))
            else:
                where_parts.append(f"{_as_float_sql(column, backend)} {operator} ?")
                params.append(float(str(value).replace(",", ".")))
        elif operator == "!=":
            where_parts.append(f"LOWER(CAST({column} AS VARCHAR)) <> ?")
            params.append(str(value).strip().lower())
        elif operator == "like":
            where_parts.append(f"LOWER(CAST({column} AS VARCHAR)) LIKE ?")
            params.append(f"%{str(value).strip().lower()}%")
        elif operator == "between":
            start, end = [part.strip() for part in str(value).split("|", 1)]
            where_parts.append(f"{_as_date_sql(column, backend)} BETWEEN ? AND ?")
            params.extend([start, end])
        elif operator == "in":
            values = [item.strip().lower() for item in str(value).split("|") if item.strip()]
            placeholders = ", ".join(["?"] * len(values))
            where_parts.append(f"LOWER(CAST({column} AS VARCHAR)) IN ({placeholders})")
            params.extend(values)
        else:
            where_parts.append(f"LOWER(CAST({column} AS VARCHAR)) = ?")
            params.append(str(value).strip().lower())
    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    intent = plan.get("intent")
    aggregation = plan.get("aggregation")
    metric = plan.get("metric_column")
    limit = int(plan.get("limit") or 10)

    if intent == "tabular_schema":
        return ("", [])

    if intent == "tabular_distinct":
        dimension = plan.get("dimension_column")
        sql = (
            f"SELECT DISTINCT CAST({dimension} AS VARCHAR) AS value FROM {table} {where_clause} "
            f"AND CAST({dimension} AS VARCHAR) <> ''" if where_clause else
            f"SELECT DISTINCT CAST({dimension} AS VARCHAR) AS value FROM {table} WHERE CAST({dimension} AS VARCHAR) <> ''"
        )
        sql += " ORDER BY value ASC LIMIT ?"
        return sql, [*params, limit]

    if intent in {"tabular_count", "tabular_aggregate"}:
        if aggregation == "count":
            return f"SELECT COUNT(*) AS value FROM {table} {where_clause}", params
        return f"SELECT {aggregation.upper()}({_as_float_sql(metric, backend)}) AS value FROM {table} {where_clause}", params

    if intent in {"tabular_groupby", "tabular_compare"}:
        group_by = [col for col in plan.get("group_by", []) if col]
        time_grain = plan.get("time_grain")
        group_exprs: list[str] = []
        select_group_exprs: list[str] = []
        for col in group_by:
            if time_grain:
                bucket_expr = _date_bucket_sql(col, backend, time_grain)
                group_exprs.append(bucket_expr)
                select_group_exprs.append(f"{bucket_expr} AS {col}")
            else:
                group_exprs.append(col)
                select_group_exprs.append(col)
        group_clause = ", ".join(group_exprs)
        select_group_clause = ", ".join(select_group_exprs)
        metric_expr = "COUNT(*)" if aggregation == "count" or not metric else f"{aggregation.upper()}({_as_float_sql(metric, backend)})"
        order_clause = ", ".join(group_exprs) if time_grain else "value DESC NULLS LAST"
        sql = (
            f"SELECT {select_group_clause}, {metric_expr} AS value FROM {table} {where_clause} "
            f"GROUP BY {group_clause} ORDER BY {order_clause} LIMIT ?"
        )
        return sql, [*params, limit]

    if intent == "catalog_filter":
        select_cols = "doc_id, row_index, page_number, raw_row, texto_canonico, *"
        sql = f"SELECT {select_cols} FROM {table} {where_clause} LIMIT ?"
        return sql, [*params, limit]

    if intent in {"catalog_deadline_report", "catalog_sla_alert"}:
        group_by = [col for col in plan.get("group_by", []) if col]
        if intent == "catalog_deadline_report" and group_by:
            group_clause = ", ".join(group_by)
            sql = (
                f"SELECT {group_clause}, COUNT(*) AS value FROM {table} {where_clause} "
                f"GROUP BY {group_clause} ORDER BY value DESC, {group_clause} ASC LIMIT ?"
            )
            return sql, [*params, limit]
        select_cols = "doc_id, row_index, page_number, raw_row, texto_canonico, *"
        sql = f"SELECT {select_cols} FROM {table} {where_clause} LIMIT ?"
        return sql, [*params, limit]

    if intent == "tabular_rank":
        display_cols = [col for col in ["nome", "name", "id_cliente", "id"]]
        select_cols = ", ".join(display_cols + [f"{_as_float_sql(metric, backend)} AS value"])
        order_clause = "DESC" if aggregation != "min" else "ASC"
        sql = f"SELECT {select_cols} FROM {table} {where_clause} ORDER BY value {order_clause} NULLS LAST LIMIT ?"
        return sql, [*params, limit]

    if intent == "tabular_describe_column":
        return ("", [])

    return ("", [])
