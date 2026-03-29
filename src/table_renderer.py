"""Renderer for semantic table-first answers."""
from __future__ import annotations

from src.table_semantics import aggregation_lead_text, infer_subject_label, render_value_by_unit


def render_table_answer(question: str, plan: dict, result: dict, context_hint: str, business_scope: str) -> tuple[str, str]:
    intent = plan.get("intent")
    if not intent:
        op = str(plan.get("operation") or result.get("operation") or "")
        intent = {
            "aggregate": "tabular_count" if plan.get("aggregation") == "count" else "tabular_aggregate",
            "groupby": "tabular_groupby",
            "rank": "tabular_rank",
            "distinct": "tabular_distinct",
            "schema": "tabular_schema",
            "describe_column": "tabular_describe_column",
            "compare": "tabular_compare",
        }.get(op, "")
    assumptions = list(plan.get("assumptions", []) or [])
    assumption_text = f" Criterio adotado: {'; '.join(assumptions)}." if assumptions else ""

    if intent in {"tabular_count", "tabular_aggregate"}:
        aggregation = plan.get("aggregation", "sum")
        if aggregation == "count":
            subject = infer_subject_label(question, context_hint)
            answer = f"A quantidade de {subject}{(' ' + business_scope) if business_scope else ''} e {render_value_by_unit(result.get('value'), 'count')}."
            return answer + assumption_text, answer
        metric_label = str(plan.get("metric_column") or "resultado").replace("_", " ")
        unit = str(plan.get("metric_unit") or "number")
        lead = aggregation_lead_text(aggregation, metric_label, unit, business_scope or "")
        answer = f"{lead} {render_value_by_unit(result.get('value'), unit, aggregation)}."
        return answer + assumption_text, answer

    if intent == "tabular_distinct":
        values = [str(v).strip() for v in result.get("values", []) if str(v).strip()]
        dimension = str(plan.get("dimension_column") or "valores").replace("_", " ")
        if not values:
            answer = f"Nao encontrei valores distintos para {dimension}."
            return answer, answer
        preview = ", ".join(values[:20])
        answer = f"Os {dimension if dimension.endswith('s') else dimension + 's'} presentes na base{(' ' + business_scope) if business_scope else ''} sao: {preview}."
        if len(values) > 20:
            answer += f" Total de {len(values)} valores distintos."
        return answer + assumption_text, preview

    if intent == "tabular_schema":
        profiles = result.get("profiles", []) or []
        columns = [item.get("name", "") for item in profiles] or result.get("columns", [])
        if not columns:
            answer = "Nao encontrei colunas disponiveis nesta tabela."
            return answer, answer
        preview = ", ".join(columns[:20])
        answer = f"As colunas da tabela sao: {preview}."
        metrics = [item["name"] for item in profiles if item.get("role") == "metric"][:6]
        dimensions = [item["name"] for item in profiles if item.get("role") == "dimension"][:6]
        if metrics:
            answer += f" Metricas principais: {', '.join(metrics)}."
        if dimensions:
            answer += f" Dimensoes principais: {', '.join(dimensions)}."
        if len(columns) > 20:
            answer += f" Total de {len(columns)} colunas."
        return answer, preview

    if intent == "tabular_describe_column":
        target = plan.get("target_column")
        descriptions = {item.get("name"): item for item in result.get("profiles", []) or []}
        profile = descriptions.get(target)
        if not profile:
            answer = "Nao encontrei detalhes semanticos para essa coluna."
            return answer, answer
        answer = (
            f"A coluna {target} representa: {profile.get('description', 'sem descricao')} "
            f"Tipo semantico: {profile.get('semantic_type', '')}. "
            f"Unidade: {profile.get('unit', '')}."
        )
        return answer, answer

    if intent == "tabular_compare":
        rows = result.get("rows", [])
        if not rows:
            answer = f"Nao encontrei linhas tabulares{(' ' + business_scope) if business_scope else ''}."
            return answer, answer
        group_by = [str(col) for col in plan.get("group_by", []) if str(col).strip()]
        group_label = ", ".join(col.replace("_", " ") for col in group_by) if group_by else "grupo"
        metric_label = str(plan.get("metric_column") or "resultado").replace("_", " ")
        aggregation = str(plan.get("aggregation") or "avg")
        unit = str(plan.get("metric_unit") or "number")
        top_rows = rows[:3]
        preview = "; ".join(
            f"{' | '.join(str(row.get(col, '')) for col in group_by)}: {render_value_by_unit(row.get('value'), unit, aggregation)}"
            for row in top_rows
        )
        if len(top_rows) >= 2:
            leader = top_rows[0]
            runner_up = top_rows[1]
            leader_label = " | ".join(str(leader.get(col, "")) for col in group_by)
            runner_label = " | ".join(str(runner_up.get(col, "")) for col in group_by)
            try:
                delta = float(leader.get("value") or 0) - float(runner_up.get("value") or 0)
                delta_text = render_value_by_unit(delta, unit, aggregation)
                answer = (
                    f"Comparativo de {metric_label} por {group_label}{(' ' + business_scope) if business_scope else ''}: "
                    f"{leader_label} lidera com {render_value_by_unit(leader.get('value'), unit, aggregation)}, "
                    f"seguido de {runner_label} com {render_value_by_unit(runner_up.get('value'), unit, aggregation)}. "
                    f"Diferenca de {delta_text}."
                )
            except Exception:
                answer = f"Comparativo de {metric_label} por {group_label}{(' ' + business_scope) if business_scope else ''}: {preview}."
        else:
            answer = f"Comparativo de {metric_label} por {group_label}{(' ' + business_scope) if business_scope else ''}: {preview}."
        return answer + assumption_text, preview

    if intent == "tabular_groupby":
        rows = result.get("rows", [])
        if not rows:
            answer = f"Nao encontrei linhas tabulares{(' ' + business_scope) if business_scope else ''}."
            return answer, answer
        group_by = [str(col) for col in plan.get("group_by", []) if str(col).strip()]
        group_label = ", ".join(col.replace("_", " ") for col in group_by) if group_by else "grupo"
        unit = str(plan.get("metric_unit") or ("count" if plan.get("aggregation") == "count" else "number"))
        aggregation = str(plan.get("aggregation") or "")
        preview = "; ".join(
            f"{' | '.join(str(row.get(col, '')) for col in group_by)}: {render_value_by_unit(row.get('value'), unit, aggregation)}"
            for row in rows[:5]
        )
        answer = f"Principais resultados por {group_label}{(' ' + business_scope) if business_scope else ''}: {preview}."
        return answer + assumption_text, preview

    if intent == "tabular_rank":
        rows = result.get("rows", [])
        if not rows:
            answer = f"Nao encontrei linhas tabulares{(' ' + business_scope) if business_scope else ''}."
            return answer, answer
        unit = str(plan.get("metric_unit") or "number")
        aggregation = str(plan.get("aggregation") or "")
        preview_cols = [col for col in rows[0].keys() if col != "value"]
        preview = "; ".join(
            ", ".join(f"{col}={row.get(col)}" for col in preview_cols[:3]) + f", valor={render_value_by_unit(row.get('value'), unit, aggregation)}"
            for row in rows[:5]
        )
        answer = f"Principais registros{(' ' + business_scope) if business_scope else ''}: {preview}."
        return answer + assumption_text, preview

    rows = result.get("rows", [])
    if not rows:
        answer = f"Nao encontrei linhas tabulares{(' ' + business_scope) if business_scope else ''}."
        return answer, answer
    preview = "; ".join(str(row) for row in rows[:5])
    return preview + assumption_text, preview
