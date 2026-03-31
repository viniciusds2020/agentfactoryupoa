"""Structured planner for semantic table-first queries."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import get_settings
from src.table_semantics import infer_table_type, normalize_semantic_text
from src.utils import get_logger, log_event

logger = get_logger(__name__)


class PlanFilter(BaseModel):
    column: str
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "in", "between"] = "="
    value: str


class QueryPlan(BaseModel):
    intent: Literal[
        "tabular_aggregate",
        "tabular_count",
        "tabular_groupby",
        "tabular_rank",
        "tabular_distinct",
        "tabular_schema",
        "tabular_describe_column",
        "tabular_compare",
        "catalog_lookup_by_id",
        "catalog_lookup_by_title",
        "catalog_field_lookup",
        "catalog_record_summary",
        "catalog_compare",
    ]
    subject: str = ""
    metric_column: str | None = None
    target_column: str | None = None
    dimension_column: str | None = None
    aggregation: Literal["sum", "avg", "count", "min", "max"] | None = None
    filters: list[PlanFilter] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    limit: int = 1
    time_grain: Literal["day", "month", "year"] | None = None
    expected_unit: str = ""
    confidence: float = 0.0
    assumptions: list[str] = Field(default_factory=list)
    planner_source: str = "heuristic"
    table_mode: str = "analytic"


def _catalog_identifier_profile(profiles: list[dict]):
    return next((p for p in profiles if p.get("role") == "identifier"), None)


def _catalog_title_profile(profiles: list[dict]):
    return next((p for p in profiles if p.get("semantic_type") == "catalog_title"), None)


def _catalog_field_profile(question: str, profiles: list[dict]):
    q_norm = normalize_semantic_text(question).replace("_", " ")
    preferred_semantic = {
        "cobertura": "coverage_rule",
        "autorizacao": "authorization_rule",
        "autoriza": "authorization_rule",
        "prazo": "deadline_rule",
    }
    for token, semantic_type in preferred_semantic.items():
        if token in q_norm:
            return next((p for p in profiles if p.get("semantic_type") == semantic_type), None)
    for profile in profiles:
        aliases = [alias for alias in profile.get("aliases", []) if alias]
        if any(re.search(rf"\b{re.escape(alias)}\b", q_norm) for alias in aliases):
            return profile
    return None


def _extract_catalog_title_value(question: str) -> str:
    patterns = [
        r"\bcodigo d[ao]s?\s+(.+?)\??$",
        r"\bqual e o codigo d[ao]s?\s+(.+?)\??$",
        r"\bqual o codigo d[ao]s?\s+(.+?)\??$",
    ]
    normalized = question.strip()
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return match.group(1).strip(" ?.")
    return ""


def _extract_catalog_ids(question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b\d{4,}\b", question)))


def _catalog_plan(question: str, profiles: list[dict], filters: list[dict], context_hint: str = "") -> QueryPlan | None:
    identifier = _catalog_identifier_profile(profiles)
    title = _catalog_title_profile(profiles)
    field_profile = _catalog_field_profile(question, profiles)
    ids = _extract_catalog_ids(question)
    q_norm = normalize_semantic_text(question).replace("_", " ")

    if re.search(r"\b(compare|comparar|comparacao|versus|vs\.?)\b", q_norm) and len(ids) >= 2 and identifier:
        return QueryPlan(
            intent="catalog_compare",
            metric_column=identifier["name"],
            filters=[PlanFilter(column=identifier["name"], operator="in", value="|".join(ids[:5]))],
            confidence=0.93,
            planner_source="heuristic",
            table_mode="catalog",
        )

    if re.search(r"\b(resuma|resumo|explique|detalhe)\b", q_norm) and ids and identifier:
        return QueryPlan(
            intent="catalog_record_summary",
            filters=[PlanFilter(column=identifier["name"], operator="=", value=ids[0])],
            confidence=0.95,
            planner_source="heuristic",
            table_mode="catalog",
        )

    if ids and identifier and field_profile and field_profile["name"] != identifier["name"]:
        return QueryPlan(
            intent="catalog_field_lookup",
            target_column=field_profile["name"],
            filters=[PlanFilter(column=identifier["name"], operator="=", value=ids[0])],
            confidence=0.97,
            planner_source="heuristic",
            table_mode="catalog",
        )

    if ids and identifier:
        return QueryPlan(
            intent="catalog_lookup_by_id",
            filters=[PlanFilter(column=identifier["name"], operator="=", value=ids[0])],
            confidence=0.97,
            planner_source="heuristic",
            table_mode="catalog",
        )

    title_value = _extract_catalog_title_value(question)
    if title and title_value:
        intent = "catalog_field_lookup" if field_profile and field_profile["name"] != title["name"] else "catalog_lookup_by_title"
        return QueryPlan(
            intent=intent,  # type: ignore[arg-type]
            target_column=field_profile["name"] if intent == "catalog_field_lookup" and field_profile else None,
            filters=[PlanFilter(column=title["name"], operator="=", value=title_value)],
            confidence=0.9,
            planner_source="heuristic",
            table_mode="catalog",
        )

    return None


def _heuristic_plan(question: str, profiles: list[dict], filters: list[dict], context_hint: str = "") -> QueryPlan | None:
    from src.structured_store import _resolve_inventory_column, _resolve_metric_column

    q_norm = normalize_semantic_text(question).replace("_", " ")
    table_mode = infer_table_type(profiles, context_hint=context_hint)
    if table_mode == "catalog":
        catalog = _catalog_plan(question, profiles, filters, context_hint=context_hint)
        if catalog:
            return catalog

    metric = _resolve_metric_column(question, [_profile_obj(p) for p in profiles], context_hint=context_hint)
    inventory = _resolve_inventory_column(question, [_profile_obj(p) for p in profiles], context_hint=context_hint)

    if re.search(r"\b(o que significa|explique a coluna|descreva a coluna|significa a coluna)\b", q_norm):
        target = None
        for profile in profiles:
            aliases = [alias for alias in profile.get("aliases", []) if alias]
            if any(re.search(rf"\b{re.escape(alias)}\b", q_norm) for alias in aliases):
                target = profile["name"]
                break
        return QueryPlan(
            intent="tabular_describe_column",
            target_column=target,
            confidence=0.9 if target else 0.5,
            assumptions=[] if target else ["coluna-alvo nao identificada com confianca alta"],
            planner_source="heuristic",
            table_mode=table_mode,
        )

    if re.search(r"\b(coluna|colunas|campos|field|fields|schema|estrutura da tabela)\b", q_norm):
        return QueryPlan(
            intent="tabular_schema",
            confidence=0.98,
            planner_source="heuristic",
            table_mode=table_mode,
        )

    if inventory and not re.search(r"\b(total|soma|somar|media|m[eé]dia|quantos|quantas|numero de|qtd|maximo|minimo|maior|menor|top)\b", q_norm):
        return QueryPlan(
            intent="tabular_distinct",
            dimension_column=inventory.name,
            filters=[PlanFilter(**item) for item in filters],
            limit=100,
            confidence=0.95,
            planner_source="heuristic",
            table_mode=table_mode,
        )

    aggregation: str | None = None
    intent: str | None = None
    if re.search(r"\b(quantos|quantas|numero de|qtd)\b", q_norm):
        aggregation = "count"
        intent = "tabular_count"
    elif re.search(r"\b(media|m[eé]dia)\b", q_norm):
        aggregation = "avg"
        intent = "tabular_aggregate"
    elif re.search(r"\b(total|soma|somar)\b", q_norm):
        aggregation = "sum"
        intent = "tabular_aggregate"
    elif re.search(r"\b(maior|maxim[oa]|maximo)\b", q_norm):
        aggregation = "max"
        intent = "tabular_aggregate"
    elif re.search(r"\b(menor|minim[oa]|minimo)\b", q_norm):
        aggregation = "min"
        intent = "tabular_aggregate"

    group_by: list[str] = []
    time_grain: Literal["day", "month", "year"] | None = None
    date_profile = next((p for p in profiles if p.get("semantic_type") in {"date", "datetime"}), None)
    if re.search(r"\b(por mes|mensal|mes a mes)\b", q_norm):
        time_grain = "month"
    elif re.search(r"\b(por ano|anual)\b", q_norm):
        time_grain = "year"
    elif re.search(r"\b(por dia|diario|dia a dia)\b", q_norm):
        time_grain = "day"
    if time_grain and date_profile:
        group_by.append(date_profile["name"])
        intent = "tabular_groupby"
    if " por " in f" {q_norm} ":
        for profile in profiles:
            aliases = [alias for alias in profile.get("aliases", []) if alias]
            if any(re.search(rf"\bpor\s+{re.escape(alias)}\b", q_norm) for alias in aliases):
                if profile["name"] not in group_by:
                    group_by.append(profile["name"])
                break
        if group_by:
            intent = "tabular_groupby"

    rank_match = re.search(r"\btop\s+(\d{1,3})\b", q_norm)
    if rank_match and metric:
        return QueryPlan(
            intent="tabular_rank",
            metric_column=metric.name,
            aggregation=aggregation or "max",
            filters=[PlanFilter(**item) for item in filters],
            limit=int(rank_match.group(1)),
            time_grain=time_grain,
            expected_unit=getattr(metric, "unit", ""),
            confidence=0.92,
            assumptions=[] if aggregation else ["ranking inferido a partir da metrica numerica principal"],
            planner_source="heuristic",
            table_mode=table_mode,
        )

    compare_match = re.search(r"\b(compare|comparar|comparacao|versus|vs\.?)\b", q_norm)
    if compare_match and metric:
        compare_dimension = None
        if re.search(r"\b(estado|uf)\b", q_norm):
            compare_dimension = next((p["name"] for p in profiles if p.get("semantic_type") == "dimension_geo_state"), None)
        elif re.search(r"\b(cidade|municipio)\b", q_norm):
            compare_dimension = next((p["name"] for p in profiles if p.get("semantic_type") == "dimension_geo_city"), None)
        elif inventory:
            compare_dimension = inventory.name
        else:
            compare_dimension = next(
                (p["name"] for p in profiles if p.get("role") == "dimension"),
                None,
            )
        return QueryPlan(
            intent="tabular_compare",
            metric_column=metric.name,
            aggregation=aggregation or "avg",
            filters=[PlanFilter(**item) for item in filters],
            group_by=[compare_dimension] if compare_dimension else [],
            limit=10,
            time_grain=time_grain,
            expected_unit=getattr(metric, "unit", ""),
            confidence=0.78,
            assumptions=["comparacao inferida pela metrica principal"],
            planner_source="heuristic",
            table_mode=table_mode,
        )

    if intent:
        return QueryPlan(
            intent=intent,  # type: ignore[arg-type]
            metric_column=metric.name if metric else None,
            aggregation=aggregation,  # type: ignore[arg-type]
            filters=[PlanFilter(**item) for item in filters],
            group_by=group_by,
            limit=10 if group_by else 1,
            time_grain=time_grain,
            expected_unit=getattr(metric, "unit", "count" if aggregation == "count" else ""),
            confidence=0.9 if metric or aggregation == "count" else 0.6,
            assumptions=[],
            planner_source="heuristic",
            table_mode=table_mode,
        )

    if metric:
        inferred_agg = "sum" if getattr(metric, "unit", "") == "brl" else "avg"
        assumptions = ["agregacao inferida pela metrica principal"]
        return QueryPlan(
            intent="tabular_aggregate",
            metric_column=metric.name,
            aggregation=inferred_agg,
            filters=[PlanFilter(**item) for item in filters],
            group_by=[],
            limit=1,
            time_grain=time_grain,
            expected_unit=getattr(metric, "unit", ""),
            confidence=0.72,
            assumptions=assumptions,
            planner_source="heuristic",
            table_mode=table_mode,
        )

    return None


def _profile_obj(profile: dict):
    from src.structured_store import ColumnProfile

    return ColumnProfile(
        profile["name"],
        profile.get("data_type", profile.get("physical_type", "text")),
        profile["role"],
        profile["semantic_type"],
        profile.get("unit", "text"),
        profile.get("aliases", []),
        profile.get("examples", []),
        profile.get("description", ""),
    )


def try_llm_plan(question: str, profiles: list[dict], context_hint: str = "") -> QueryPlan | None:
    settings = get_settings()
    if settings.llm_provider not in {"groq", "anthropic"}:
        return None
    if not getattr(settings, "table_planner_llm_enabled", False):
        return None
    from src import llm

    schema_summary = json.dumps(
        [
            {
                "name": p["name"],
                "role": p["role"],
                "semantic_type": p["semantic_type"],
                "unit": p["unit"],
                "aliases": p.get("aliases", [])[:6],
            }
            for p in profiles
        ],
        ensure_ascii=False,
    )
    prompt = (
        "Voce e um planner de consultas tabulares. Responda apenas JSON valido no schema pedido. "
        "Nao escreva explicacoes.\n"
        f"context_hint={context_hint}\n"
        f"schema={schema_summary}\n"
        f"question={question}\n"
        "Schema JSON: "
        '{"intent":"","subject":"","metric_column":null,"target_column":null,"dimension_column":null,'
        '"aggregation":null,"filters":[],"group_by":[],"order_by":[],"limit":1,"expected_unit":"","confidence":0.0,"assumptions":[]}'
    )
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], system="Responda somente JSON.")
        payload = json.loads(raw)
        plan = QueryPlan.model_validate(payload)
        plan.planner_source = f"llm:{settings.llm_provider}"
        return plan
    except Exception as exc:
        log_event(logger, 30, "LLM table planner failed; falling back to heuristic", error=str(exc))
        return None


def build_query_plan(question: str, profiles: list[dict], filters: list[dict], context_hint: str = "") -> QueryPlan | None:
    heuristic = _heuristic_plan(question, profiles, filters, context_hint=context_hint)
    if heuristic and heuristic.confidence >= 0.85:
        return heuristic
    llm_plan = try_llm_plan(question, profiles, context_hint=context_hint)
    if llm_plan:
        return llm_plan
    return heuristic
