"""Semantic helpers for generic tabular understanding and rendering."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


SemanticType = Literal[
    "identifier",
    "catalog_title",
    "catalog_attribute",
    "coverage_rule",
    "authorization_rule",
    "deadline_rule",
    "dimension_person",
    "dimension_organization",
    "dimension_geo_state",
    "dimension_geo_city",
    "dimension_status",
    "dimension_category",
    "flag_boolean",
    "measure_currency",
    "measure_age_years",
    "measure_score",
    "measure_count",
    "measure_percentage",
    "measure_number",
    "date",
    "datetime",
    "text",
]

UnitType = Literal["brl", "anos", "pontos", "count", "number", "percent", "boolean", "date", "datetime", "text"]


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
    if re.search(r"(^id$|^id_|_id$|codigo|code|chave|key|procedimento|matricula|registro)", name):
        return SemanticColumn("identifier", "text", "Identificador da linha ou entidade.")
    if re.search(r"(data|date)", name):
        return SemanticColumn("date", "date", "Data ou referencia temporal.")
    if re.search(r"(hora|timestamp|datetime|criado_em|updated_at)", name):
        return SemanticColumn("datetime", "datetime", "Data e hora do evento ou atualizacao.")
    if re.search(r"(descricao|descr|titulo|title|nome_item|nome_produto|produto_nome|item_nome|servico|service|sku_name)", name):
        return SemanticColumn("catalog_title", "text", "Titulo ou descricao principal do registro.")
    if re.search(r"(cobertura)", name):
        return SemanticColumn("coverage_rule", "text", "Regra de cobertura do item ou procedimento.")
    if re.search(r"(autoriz)", name):
        return SemanticColumn("authorization_rule", "text", "Regra ou orientacao de autorizacao.")
    if re.search(r"(prazo|deadline)", name):
        return SemanticColumn("deadline_rule", "text", "Prazo ou janela operacional associada ao registro.")
    if re.search(r"(estado|uf)", name):
        return SemanticColumn("dimension_geo_state", "text", "Dimensao geografica de unidade federativa.")
    if re.search(r"(cidade|municipio)", name):
        return SemanticColumn("dimension_geo_city", "text", "Dimensao geografica de cidade.")
    if re.search(r"(cliente|pessoa|usuario|paciente|beneficiario|cooperado|aluno|funcionario|colaborador|nome)", name):
        return SemanticColumn("dimension_person", "text", "Entidade ou pessoa identificada na tabela.")
    if re.search(r"(empresa|fornecedor|prestador|hospital|clinica|organizacao|instituicao)", name):
        return SemanticColumn("dimension_organization", "text", "Organizacao ou entidade de negocio.")
    if re.search(r"(status|situacao)", name):
        return SemanticColumn("dimension_status", "text", "Dimensao de status ou situacao.")
    if re.search(r"(segmento|segmentacao|categoria|tipo|canal|departamento)", name):
        return SemanticColumn("dimension_category", "text", "Dimensao categorica de negocio.")
    if re.search(r"(percent|percentual|taxa|aliquota|pct)", name):
        return SemanticColumn("measure_percentage", "percent", "Medida percentual.")
    if re.search(r"(emergencia|urgencia)", name):
        return SemanticColumn("flag_boolean", "boolean", "Indicador de atendimento de emergencia ou urgencia.")
    if re.search(r"(ativo|flag|boolean|is_|tem_|possui_)", name):
        return SemanticColumn("flag_boolean", "boolean", "Indicador booleano ou de presenca.")
    if re.search(r"(idade|age)", name):
        return SemanticColumn("measure_age_years", "anos", "Medida de idade em anos.")
    if re.search(r"(renda|salario|faturamento|receita|valor|preco|price|amount|custo)", name):
        return SemanticColumn("measure_currency", "brl", "Medida monetaria.")
    if re.search(r"(score|nota|pontuacao|pontuacao_credito|score_credito)", name):
        return SemanticColumn("measure_score", "pontos", "Score ou pontuacao.")
    if numeric_ratio >= 0.8:
        return SemanticColumn("measure_number", "number", "Medida numerica generica.")
    if re.search(r"(segmentacao|observacao|orientacao|regra|campo|descricao)", name):
        return SemanticColumn("catalog_attribute", "text", "Atributo textual do registro.")
    return SemanticColumn("text", "text", "Campo textual generico.")


def semantic_role_for_type(semantic_type: SemanticType) -> Literal["metric", "dimension", "identifier", "date", "text"]:
    if semantic_type == "identifier":
        return "identifier"
    if semantic_type == "date":
        return "date"
    if semantic_type in {"catalog_title", "catalog_attribute", "coverage_rule", "authorization_rule", "deadline_rule"}:
        return "text"
    if semantic_type.startswith("dimension_") or semantic_type == "flag_boolean":
        return "dimension"
    if semantic_type.startswith("measure_"):
        return "metric"
    return "text"


def infer_table_type(profiles: list[dict] | list[object], context_hint: str = "") -> str:
    semantic_types = [str(getattr(p, "semantic_type", "") if not isinstance(p, dict) else p.get("semantic_type", "")) for p in profiles]
    names = [normalize_semantic_text(getattr(p, "name", "") if not isinstance(p, dict) else p.get("name", "")) for p in profiles]
    context = normalize_semantic_text(context_hint).replace("_", " ")

    identifier_count = sum(1 for item in semantic_types if item == "identifier")
    title_count = sum(1 for item in semantic_types if item == "catalog_title")
    catalog_attr_count = sum(1 for item in semantic_types if item in {"catalog_attribute", "coverage_rule", "authorization_rule", "deadline_rule"})
    metric_count = sum(1 for item in semantic_types if item.startswith("measure_"))
    date_count = sum(1 for item in semantic_types if item in {"date", "datetime"})
    category_count = sum(1 for item in semantic_types if item.startswith("dimension_") or item == "flag_boolean")

    catalog_signals = {
        "procedimento",
        "descricao",
        "cobertura",
        "autorizacao",
        "prazo",
        "segmentacao",
        "rol",
        "codigo",
    }
    context_signals = any(signal in context for signal in catalog_signals)
    name_signals = sum(1 for name in names if any(signal in name for signal in catalog_signals))

    if identifier_count >= 1 and (title_count >= 1 or catalog_attr_count >= 2) and (catalog_attr_count >= 2 or metric_count <= 1):
        return "catalog"
    if context_signals and identifier_count >= 1 and name_signals >= 2:
        return "catalog"
    if date_count >= 1 and metric_count >= 1 and category_count >= 1:
        return "transactional"
    if date_count >= 1 and metric_count >= 1:
        return "time_series"
    if date_count >= 1 and category_count >= 2 and metric_count == 0:
        return "mixed"
    return "analytic"


def infer_profiles_from_records(column_names: list[str], records: list[object], sample_size: int = 50) -> list[dict]:
    sampled = list(records[:sample_size]) if records else []
    profiles: list[dict] = []
    for column_name in column_names:
        non_empty = 0
        numeric = 0
        for record in sampled:
            fields = getattr(record, "fields", {}) or {}
            value = str(fields.get(column_name, "") or "").strip()
            if not value:
                continue
            non_empty += 1
            normalized = value.replace(".", "").replace(",", ".")
            try:
                float(normalized)
                numeric += 1
            except Exception:
                pass
        numeric_ratio = (numeric / non_empty) if non_empty else 0.0
        semantic = infer_semantic_column(column_name, numeric_ratio)
        profiles.append(
            {
                "name": column_name,
                "semantic_type": semantic.semantic_type,
                "role": semantic_role_for_type(semantic.semantic_type),
                "unit": semantic.unit,
            }
        )
    return profiles


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
    if unit == "percent":
        return f"{formatted}%"
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
        ("procedimentos", "procedimentos"),
        ("procedimento", "procedimentos"),
        ("codigos", "codigos"),
        ("codigo", "codigos"),
        ("itens", "itens"),
        ("item", "itens"),
    ]
    for token, label in subject_map:
        if re.search(rf"\b{re.escape(token)}\b", haystack):
            return label
    return "registros"
