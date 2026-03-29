"""Semantic helpers for generic tabular understanding and rendering."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


SemanticType = Literal[
    "identifier",
    "dimension_geo_state",
    "dimension_geo_city",
    "dimension_status",
    "dimension_category",
    "measure_currency",
    "measure_age_years",
    "measure_score",
    "measure_count",
    "measure_number",
    "date",
    "text",
]

UnitType = Literal["brl", "anos", "pontos", "count", "number", "date", "text"]


@dataclass
class SemanticColumn:
    semantic_type: SemanticType
    unit: UnitType
    description: str


def normalize_semantic_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return normalized


def infer_semantic_column(column_name: str, numeric_ratio: float) -> SemanticColumn:
    name = normalize_semantic_text(column_name)
    if re.search(r"(^id$|^id_|_id$|codigo|code|chave|key)", name):
        return SemanticColumn("identifier", "text", "Identificador da linha ou entidade.")
    if re.search(r"(data|date)", name):
        return SemanticColumn("date", "date", "Data ou referencia temporal.")
    if re.search(r"(estado|uf)", name):
        return SemanticColumn("dimension_geo_state", "text", "Dimensao geografica de unidade federativa.")
    if re.search(r"(cidade|municipio)", name):
        return SemanticColumn("dimension_geo_city", "text", "Dimensao geografica de cidade.")
    if re.search(r"(status|situacao)", name):
        return SemanticColumn("dimension_status", "text", "Dimensao de status ou situacao.")
    if re.search(r"(segmento|categoria|tipo|canal|departamento)", name):
        return SemanticColumn("dimension_category", "text", "Dimensao categorica de negocio.")
    if re.search(r"(idade|age)", name):
        return SemanticColumn("measure_age_years", "anos", "Medida de idade em anos.")
    if re.search(r"(renda|salario|faturamento|receita|valor|preco|price|amount|custo)", name):
        return SemanticColumn("measure_currency", "brl", "Medida monetaria.")
    if re.search(r"(score|nota|pontuacao|pontuacao_credito|score_credito)", name):
        return SemanticColumn("measure_score", "pontos", "Score ou pontuacao.")
    if numeric_ratio >= 0.8:
        return SemanticColumn("measure_number", "number", "Medida numerica generica.")
    return SemanticColumn("text", "text", "Campo textual generico.")


def semantic_role_for_type(semantic_type: SemanticType) -> Literal["metric", "dimension", "identifier", "date", "text"]:
    if semantic_type == "identifier":
        return "identifier"
    if semantic_type == "date":
        return "date"
    if semantic_type.startswith("dimension_"):
        return "dimension"
    if semantic_type.startswith("measure_"):
        return "metric"
    return "text"


def render_value_by_unit(value, unit: UnitType, aggregation: str | None = None) -> str:
    if value is None:
        return "sem resultado"
    if unit == "count":
        try:
            return str(int(round(float(value))))
        except Exception:
            return str(value)

    try:
        num = float(value)
    except Exception:
        return str(value)

    formatted = f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if unit == "brl":
        return f"R$ {formatted}"
    if unit == "anos":
        return f"{formatted} anos"
    if unit == "pontos":
        return f"{formatted} pontos"
    return formatted


def aggregation_lead_text(aggregation: str, metric_label: str, unit: UnitType, scope_text: str = "") -> str:
    suffix = f" {scope_text}" if scope_text else ""
    if aggregation == "count":
        return f"A quantidade{suffix} e"
    if aggregation == "sum":
        return f"A {metric_label} total{suffix} e"
    if aggregation == "avg":
        return f"A {metric_label} media{suffix} e"
    if aggregation == "max":
        return f"O maior valor de {metric_label}{suffix} e"
    if aggregation == "min":
        return f"O menor valor de {metric_label}{suffix} e"
    return f"O resultado de {metric_label}{suffix} e"


def infer_subject_label(question: str, context_hint: str = "") -> str:
    normalized = unicodedata.normalize("NFKD", f"{question} {context_hint}")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    haystack = re.sub(r"[^a-zA-Z0-9]+", " ", normalized).strip().lower()
    subject_map = [
        ("clientes", "clientes"),
        ("cliente", "clientes"),
        ("colaboradores", "colaboradores"),
        ("colaborador", "colaboradores"),
        ("funcionarios", "funcionarios"),
        ("funcionario", "funcionarios"),
        ("pedidos", "pedidos"),
        ("pedido", "pedidos"),
        ("vendas", "vendas"),
        ("venda", "vendas"),
        ("transacoes", "transacoes"),
        ("transacao", "transacoes"),
        ("atendimentos", "atendimentos"),
        ("atendimento", "atendimentos"),
    ]
    for token, label in subject_map:
        if re.search(rf"\b{re.escape(token)}\b", haystack):
            return label
    return "registros"
