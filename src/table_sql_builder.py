"""Safe SQL builder for table-first plans."""
from __future__ import annotations


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
            where_parts.append(f"{_as_float_sql(column, backend)} {operator} ?")
            params.append(float(str(value).replace(",", ".")))
        elif operator == "!=":
            where_parts.append(f"LOWER(CAST({column} AS VARCHAR)) <> ?")
            params.append(str(value).strip().lower())
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
        group_clause = ", ".join(group_by)
        metric_expr = "COUNT(*)" if aggregation == "count" or not metric else f"{aggregation.upper()}({_as_float_sql(metric, backend)})"
        sql = (
            f"SELECT {group_clause}, {metric_expr} AS value FROM {table} {where_clause} "
            f"GROUP BY {group_clause} ORDER BY value DESC NULLS LAST LIMIT ?"
        )
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
