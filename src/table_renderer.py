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
            "catalog_lookup": "catalog_lookup_by_id",
            "catalog_field_lookup": "catalog_field_lookup",
            "catalog_record_summary": "catalog_record_summary",
            "catalog_compare": "catalog_compare",
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

    if intent in {"catalog_lookup_by_id", "catalog_lookup_by_title"}:
        record = result.get("record") or {}
        profiles = result.get("profiles", []) or []
        code = _catalog_first_value(record, profiles, {"identifier"}, ["procedimento", "codigo", "code", "id"])
        title = _catalog_first_value(record, profiles, {"catalog_title"}, ["descricao_unimed_poa", "descricao", "titulo", "title", "nome"])
        if not record:
            answer = "Nao encontrei um registro correspondente nesse catalogo."
            return answer, answer
        if code and title:
            answer = f"O codigo {code} corresponde a {title}."
        elif title:
            answer = f"Encontrei o registro {title}."
        else:
            answer = f"Encontrei o registro identificado por {code}."
        summary = _catalog_summary_suffix(record)
        if summary:
            answer += f" {summary}"
        return answer + assumption_text, title or code or answer

    if intent == "catalog_field_lookup":
        record = result.get("record") or {}
        profiles = result.get("profiles", []) or []
        target = str(plan.get("target_column") or result.get("target_column") or "").strip()
        code = _catalog_first_value(record, profiles, {"identifier"}, ["procedimento", "codigo", "code", "id"])
        title = _catalog_first_value(record, profiles, {"catalog_title"}, ["descricao_unimed_poa", "descricao", "titulo", "title", "nome"])
        value = record.get(target)
        field_label = _catalog_field_label(target, profiles)
        if value is None or value == "":
            answer = f"Nao encontrei o campo {field_label} para esse registro."
            return answer, answer
        subject = f"O procedimento {code}" if code else "O registro"
        if title:
            subject += f" ({title})"
        answer = f"{subject} possui {field_label}: {value}."
        return answer + assumption_text, str(value)

    if intent == "catalog_record_summary":
        record = result.get("record") or {}
        profiles = result.get("profiles", []) or []
        if not record:
            answer = "Nao encontrei um registro correspondente nesse catalogo."
            return answer, answer
        code = _catalog_first_value(record, profiles, {"identifier"}, ["procedimento", "codigo", "code", "id"])
        title = _catalog_first_value(record, profiles, {"catalog_title"}, ["descricao_unimed_poa", "descricao", "titulo", "title", "nome"])
        answer = f"O codigo {code} corresponde a {title}." if code and title else "Encontrei o registro solicitado."
        summary = _catalog_summary_suffix(record, profiles=profiles, full=True)
        if summary:
            answer += f" {summary}"
        return answer + assumption_text, title or code or answer

    if intent == "catalog_compare":
        rows = result.get("rows", [])
        profiles = result.get("profiles", []) or []
        if len(rows) < 2:
            answer = "Nao encontrei registros suficientes para comparar nesse catalogo."
            return answer, answer
        left = rows[0]
        right = rows[1]
        left_code = _catalog_first_value(left, profiles, {"identifier"}, ["procedimento", "codigo", "code", "id"])
        left_title = _catalog_first_value(left, profiles, {"catalog_title"}, ["descricao_unimed_poa", "descricao", "titulo", "title", "nome"])
        right_code = _catalog_first_value(right, profiles, {"identifier"}, ["procedimento", "codigo", "code", "id"])
        right_title = _catalog_first_value(right, profiles, {"catalog_title"}, ["descricao_unimed_poa", "descricao", "titulo", "title", "nome"])
        answer = (
            f"Comparativo de catalogo: {left_code} ({left_title}) versus {right_code} ({right_title}). "
            f"{_catalog_compare_sentence(left, right, profiles=profiles)}"
        )
        preview = f"{left_code}={left_title}; {right_code}={right_title}"
        return answer + assumption_text, preview

    if intent == "tabular_groupby":
        rows = result.get("rows", [])
        if not rows:
            answer = f"Nao encontrei linhas tabulares{(' ' + business_scope) if business_scope else ''}."
            return answer, answer
        group_by = [str(col) for col in plan.get("group_by", []) if str(col).strip()]
        group_label = ", ".join(col.replace("_", " ") for col in group_by) if group_by else "grupo"
        time_grain = str(plan.get("time_grain") or "").strip()
        unit = str(plan.get("metric_unit") or ("count" if plan.get("aggregation") == "count" else "number"))
        aggregation = str(plan.get("aggregation") or "")
        preview = "; ".join(
            f"{' | '.join(str(row.get(col, '')) for col in group_by)}: {render_value_by_unit(row.get('value'), unit, aggregation)}"
            for row in rows[:5]
        )
        if time_grain:
            grain_label = {"day": "dia", "month": "mes", "year": "ano"}.get(time_grain, time_grain)
            answer = f"Evolucao por {grain_label} de {group_label}{(' ' + business_scope) if business_scope else ''}: {preview}."
        else:
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


def _first_record_value(record: dict, candidates: list[str]) -> str:
    lowered = {str(key).lower(): value for key, value in (record or {}).items()}
    for candidate in candidates:
        if candidate.lower() in lowered and lowered[candidate.lower()] not in {None, ""}:
            return str(lowered[candidate.lower()])
    return ""


def _catalog_profile_names_by_semantic(profiles: list[dict], semantic_types: set[str]) -> list[str]:
    names: list[str] = []
    for profile in profiles or []:
        if str(profile.get("semantic_type", "")).strip() in semantic_types:
            name = str(profile.get("name") or profile.get("column_name") or "").strip()
            if name:
                names.append(name)
    return names


def _catalog_first_value(record: dict, profiles: list[dict], semantic_types: set[str], fallback_candidates: list[str]) -> str:
    semantic_names = _catalog_profile_names_by_semantic(profiles, semantic_types)
    value = _first_record_value(record, semantic_names)
    if value:
        return value
    return _first_record_value(record, fallback_candidates)


def _catalog_field_label(column_name: str, profiles: list[dict]) -> str:
    for profile in profiles or []:
        name = str(profile.get("name") or profile.get("column_name") or "").strip()
        if name == column_name:
            display = str(profile.get("display_name") or "").strip()
            return display or column_name.replace("_", " ")
    return column_name.replace("_", " ")


def _catalog_summary_suffix(record: dict, profiles: list[dict] | None = None, full: bool = False) -> str:
    profiles = profiles or []
    coverage = _catalog_first_value(record, profiles, {"coverage_rule"}, ["cobertura_unimed_poa", "cobertura"])
    segmentation = _first_record_value(record, _catalog_profile_names_by_semantic(profiles, {"catalog_attribute"})) or _first_record_value(record, ["segmentacao_ans", "segmentacao"])
    emergency = _first_record_value(record, ["emergencia"])
    deadline = _catalog_first_value(record, profiles, {"deadline_rule"}, ["prazo_autorizacao_conforme_rn_n_623_ans", "prazo_autorizacao", "prazo"])
    authorization = _catalog_first_value(record, profiles, {"authorization_rule"}, ["orientacao_autorizacao_call_center", "orientacao_autorizacao", "autorizacao"])
    parts: list[str] = []
    if coverage:
        parts.append(f"Cobertura: {coverage}.")
    if segmentation and full:
        parts.append(f"Segmentacao: {segmentation}.")
    if emergency:
        parts.append(f"Emergencia: {emergency}.")
    if deadline:
        parts.append(f"Prazo: {deadline}.")
    if authorization:
        parts.append(f"Autorizacao: {authorization}.")
    return " ".join(parts)


def _catalog_compare_sentence(left: dict, right: dict, profiles: list[dict] | None = None) -> str:
    profiles = profiles or []
    shared_fields = [
        (
            "cobertura",
            _catalog_first_value(left, profiles, {"coverage_rule"}, ["cobertura_unimed_poa", "cobertura"]),
            _catalog_first_value(right, profiles, {"coverage_rule"}, ["cobertura_unimed_poa", "cobertura"]),
        ),
        (
            "prazo",
            _catalog_first_value(left, profiles, {"deadline_rule"}, ["prazo_autorizacao_conforme_rn_n_623_ans", "prazo_autorizacao", "prazo"]),
            _catalog_first_value(right, profiles, {"deadline_rule"}, ["prazo_autorizacao_conforme_rn_n_623_ans", "prazo_autorizacao", "prazo"]),
        ),
        (
            "autorizacao",
            _catalog_first_value(left, profiles, {"authorization_rule"}, ["orientacao_autorizacao_call_center", "orientacao_autorizacao", "autorizacao"]),
            _catalog_first_value(right, profiles, {"authorization_rule"}, ["orientacao_autorizacao_call_center", "orientacao_autorizacao", "autorizacao"]),
        ),
    ]
    differences = [f"{label}: '{left_value}' vs '{right_value}'" for label, left_value, right_value in shared_fields if left_value and right_value and left_value != right_value]
    if differences:
        return "Principais diferencas: " + "; ".join(differences[:3]) + "."
    return "Os dois registros compartilham regras semelhantes nos principais atributos disponiveis."
