"""Structured planner for semantic table-first queries."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import get_settings
from src.table_semantics import normalize_semantic_text
from src.utils import get_logger, log_event

logger = get_logger(__name__)


class PlanFilter(BaseModel):
    column: str
    operator: Literal["=", "!=", ">", "<", ">=", "<=", "in"] = "="
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
    expected_unit: str = ""
    confidence: float = 0.0
    assumptions: list[str] = Field(default_factory=list)
    planner_source: str = "heuristic"


def _heuristic_plan(question: str, profiles: list[dict], filters: list[dict], context_hint: str = "") -> QueryPlan | None:
    from src.structured_store import _resolve_inventory_column, _resolve_metric_column

    q_norm = normalize_semantic_text(question).replace("_", " ")
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
        )

    if re.search(r"\b(coluna|colunas|campos|field|fields|schema|estrutura da tabela)\b", q_norm):
        return QueryPlan(
            intent="tabular_schema",
            confidence=0.98,
            planner_source="heuristic",
        )

    if inventory and not re.search(r"\b(total|soma|somar|media|m[eé]dia|quantos|quantas|numero de|qtd|maximo|minimo|maior|menor|top)\b", q_norm):
        return QueryPlan(
            intent="tabular_distinct",
            dimension_column=inventory.name,
            filters=[PlanFilter(**item) for item in filters],
            limit=100,
            confidence=0.95,
            planner_source="heuristic",
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
    if " por " in f" {q_norm} ":
        for profile in profiles:
            aliases = [alias for alias in profile.get("aliases", []) if alias]
            if any(re.search(rf"\bpor\s+{re.escape(alias)}\b", q_norm) for alias in aliases):
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
            expected_unit=getattr(metric, "unit", ""),
            confidence=0.92,
            assumptions=[] if aggregation else ["ranking inferido a partir da metrica numerica principal"],
            planner_source="heuristic",
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
            expected_unit=getattr(metric, "unit", ""),
            confidence=0.78,
            assumptions=["comparacao inferida pela metrica principal"],
            planner_source="heuristic",
        )

    if intent:
        return QueryPlan(
            intent=intent,  # type: ignore[arg-type]
            metric_column=metric.name if metric else None,
            aggregation=aggregation,  # type: ignore[arg-type]
            filters=[PlanFilter(**item) for item in filters],
            group_by=group_by,
            limit=10 if group_by else 1,
            expected_unit=getattr(metric, "unit", "count" if aggregation == "count" else ""),
            confidence=0.9 if metric or aggregation == "count" else 0.6,
            assumptions=[],
            planner_source="heuristic",
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
            expected_unit=getattr(metric, "unit", ""),
            confidence=0.72,
            assumptions=assumptions,
            planner_source="heuristic",
        )

    return None


def _profile_obj(profile: dict):
    from src.structured_store import ColumnProfile

    return ColumnProfile(
        profile["name"],
        profile["data_type"],
        profile["role"],
        profile["semantic_type"],
        profile["unit"],
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
