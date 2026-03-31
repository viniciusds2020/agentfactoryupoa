"""Validation layer for semantic table query plans."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    valid: bool
    normalized_plan: dict
    errors: list[str] = Field(default_factory=list)


_ALLOWED_BY_SEMANTIC = {
    "measure_currency": {"sum", "avg", "min", "max"},
    "measure_age_years": {"avg", "min", "max"},
    "measure_score": {"avg", "min", "max"},
    "measure_percentage": {"avg", "min", "max"},
    "measure_number": {"sum", "avg", "min", "max"},
    "identifier": set(),
    "catalog_title": set(),
    "catalog_attribute": set(),
    "coverage_rule": set(),
    "authorization_rule": set(),
    "deadline_rule": set(),
    "dimension_person": set(),
    "dimension_organization": set(),
    "dimension_geo_state": set(),
    "dimension_geo_city": set(),
    "dimension_status": set(),
    "dimension_category": set(),
    "flag_boolean": set(),
    "text": set(),
    "date": set(),
    "datetime": set(),
}


def validate_query_plan(plan: dict, profiles: list[dict]) -> ValidationResult:
    profile_map = {item["name"]: item for item in profiles}
    errors: list[str] = []
    normalized = dict(plan)
    intent = str(plan.get("intent", ""))
    metric = plan.get("metric_column")
    aggregation = plan.get("aggregation")
    dimension = plan.get("dimension_column")
    target_column = plan.get("target_column")

    if intent in {"tabular_aggregate", "tabular_count", "tabular_groupby", "tabular_rank", "tabular_compare"}:
        if intent != "tabular_count" and not metric:
            errors.append("metric_column obrigatoria para esta intencao")
        if metric and metric not in profile_map:
            errors.append(f"metric_column desconhecida: {metric}")
        if metric and metric in profile_map:
            semantic_type = profile_map[metric]["semantic_type"]
            if aggregation and aggregation not in _ALLOWED_BY_SEMANTIC.get(semantic_type, set()) and aggregation != "count":
                errors.append(f"agregacao {aggregation} invalida para semantic_type {semantic_type}")
            normalized["metric_unit"] = profile_map[metric]["unit"]
            normalized["metric_semantic_type"] = semantic_type

    if intent in {"tabular_groupby", "tabular_compare"}:
        for group_col in plan.get("group_by", []) or []:
            if group_col not in profile_map:
                errors.append(f"group_by desconhecida: {group_col}")
        time_grain = plan.get("time_grain")
        if time_grain and not any(profile_map.get(group_col, {}).get("semantic_type") in {"date", "datetime"} for group_col in plan.get("group_by", []) or []):
            errors.append("time_grain requer group_by com coluna de data")

    if intent == "tabular_count":
        normalized["aggregation"] = "count"
        normalized["metric_unit"] = "count"
        normalized["metric_semantic_type"] = "measure_count"

    if intent == "tabular_distinct":
        if not dimension:
            errors.append("dimension_column obrigatoria para distinct")
        elif dimension not in profile_map:
            errors.append(f"dimension_column desconhecida: {dimension}")

    if intent == "tabular_schema":
        normalized["metric_column"] = None
        normalized["aggregation"] = None

    if intent == "tabular_describe_column":
        if not target_column:
            errors.append("target_column obrigatoria para describe_column")
        elif target_column not in profile_map:
            errors.append(f"target_column desconhecida: {target_column}")

    if intent in {"catalog_lookup_by_id", "catalog_lookup_by_title", "catalog_record_summary", "catalog_compare"}:
        if not (plan.get("filters") or []):
            errors.append("filtros obrigatorios para lookup/catalogo")

    if intent == "catalog_field_lookup":
        if not target_column:
            errors.append("target_column obrigatoria para catalog_field_lookup")
        elif target_column not in profile_map:
            errors.append(f"target_column desconhecida: {target_column}")
        if not (plan.get("filters") or []):
            errors.append("filtros obrigatorios para catalog_field_lookup")

    for flt in plan.get("filters", []) or []:
        column = flt.get("column")
        if column and column not in profile_map:
            errors.append(f"filter column desconhecida: {column}")

    confidence = float(plan.get("confidence", 0.0) or 0.0)
    normalized["confidence"] = max(0.0, min(confidence, 1.0))
    normalized["assumptions"] = list(plan.get("assumptions", []) or [])
    return ValidationResult(valid=not errors, normalized_plan=normalized, errors=errors)
