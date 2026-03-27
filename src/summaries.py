"""Pre-computed summaries for legal document structural nodes.

Generates and stores summaries for chapters, sections, and titles via LLM.
Summaries include: executive summary, legal summary, key points, covered articles,
obligations, restrictions, and definitions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.legal_tree import LegalNode, LegalTree
from src.utils import get_logger

logger = get_logger(__name__)

# Maximum text length sent to LLM for summarization (chars).
# Longer texts are truncated with a note.
_MAX_SUMMARY_INPUT = 12000

_SUMMARY_SYSTEM_PROMPT = """\
Voce e um analista juridico especializado em documentos corporativos brasileiros.
Sua tarefa e gerar um resumo estruturado de uma secao de documento juridico.

REGRAS:
- Baseie-se APENAS no texto fornecido.
- Se o texto for insuficiente para um campo, escreva "Nao identificado no trecho".
- Seja conciso e objetivo.
- Mantenha termos juridicos quando relevantes.
- Responda APENAS com o JSON solicitado, sem texto adicional.
"""

_SUMMARY_USER_TEMPLATE = """\
Secao: {label}
Caminho hierarquico: {path}
Artigos contidos: {articles}

Texto da secao:
{text}

Gere um JSON com exatamente estes campos:
{{
  "resumo_executivo": "Resumo em 2-3 frases para leitura rapida por nao-juristas",
  "resumo_juridico": "Resumo tecnico-juridico em 3-5 frases com terminologia precisa",
  "pontos_chave": ["ponto 1", "ponto 2", ...],
  "artigos_cobertos": ["Art. 1", "Art. 2", ...],
  "obrigacoes": ["obrigacao 1", ...],
  "restricoes": ["restricao/vedacao 1", ...],
  "definicoes": ["termo: definicao", ...]
}}
"""


@dataclass
class NodeSummary:
    """Pre-computed summary for a legal document node."""
    node_id: str
    node_type: str
    label: str
    path: str
    resumo_executivo: str = ""
    resumo_juridico: str = ""
    pontos_chave: list[str] = field(default_factory=list)
    artigos_cobertos: list[str] = field(default_factory=list)
    obrigacoes: list[str] = field(default_factory=list)
    restricoes: list[str] = field(default_factory=list)
    definicoes: list[str] = field(default_factory=list)
    text_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "path": self.path,
            "resumo_executivo": self.resumo_executivo,
            "resumo_juridico": self.resumo_juridico,
            "pontos_chave": self.pontos_chave,
            "artigos_cobertos": self.artigos_cobertos,
            "obrigacoes": self.obrigacoes,
            "restricoes": self.restricoes,
            "definicoes": self.definicoes,
            "text_length": self.text_length,
        }

    @staticmethod
    def from_dict(data: dict) -> NodeSummary:
        return NodeSummary(
            node_id=data.get("node_id", ""),
            node_type=data.get("node_type", ""),
            label=data.get("label", ""),
            path=data.get("path", ""),
            resumo_executivo=data.get("resumo_executivo", ""),
            resumo_juridico=data.get("resumo_juridico", ""),
            pontos_chave=data.get("pontos_chave", []),
            artigos_cobertos=data.get("artigos_cobertos", []),
            obrigacoes=data.get("obrigacoes", []),
            restricoes=data.get("restricoes", []),
            definicoes=data.get("definicoes", []),
            text_length=data.get("text_length", 0),
        )


def _truncate_text(text: str, max_chars: int = _MAX_SUMMARY_INPUT) -> str:
    """Truncate text with note if too long."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... texto truncado por limite de contexto ...]"


def _parse_llm_json(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = response.strip()
    # Remove markdown code block wrappers
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (``` markers)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse LLM summary JSON, returning empty")
        return {}


def generate_node_summary(node: LegalNode) -> NodeSummary:
    """Generate a pre-computed summary for a single legal tree node via LLM.

    Uses lazy import of llm to avoid circular dependency.
    """
    from src import llm

    text = _truncate_text(node.text)
    articles_str = ", ".join(node.articles[:50]) if node.articles else "Nenhum"

    user_msg = _SUMMARY_USER_TEMPLATE.format(
        label=node.label,
        path=node.path,
        articles=articles_str,
        text=text,
    )

    try:
        response = llm.chat(
            [{"role": "user", "content": user_msg}],
            system=_SUMMARY_SYSTEM_PROMPT,
        )
        data = _parse_llm_json(response)
    except Exception as exc:
        logger.error(f"Failed to generate summary for node {node.id}: {exc}")
        data = {}

    return NodeSummary(
        node_id=node.id,
        node_type=node.node_type,
        label=node.label,
        path=node.path,
        resumo_executivo=data.get("resumo_executivo", ""),
        resumo_juridico=data.get("resumo_juridico", ""),
        pontos_chave=data.get("pontos_chave", []),
        artigos_cobertos=data.get("artigos_cobertos", []),
        obrigacoes=data.get("obrigacoes", []),
        restricoes=data.get("restricoes", []),
        definicoes=data.get("definicoes", []),
        text_length=len(node.text),
    )


def generate_tree_summaries(
    tree: LegalTree,
    node_types: set[str] | None = None,
) -> list[NodeSummary]:
    """Generate summaries for all macro nodes of a legal tree.

    Args:
        tree: The legal tree to summarize.
        node_types: Which node types to summarize. Defaults to chapters and sections.

    Returns:
        List of NodeSummary objects.
    """
    target_types = node_types or {"capitulo", "secao", "titulo"}
    nodes = [n for n in tree.node_index.values() if n.node_type in target_types]

    if not nodes:
        logger.info(f"No macro nodes found for summarization in doc '{tree.doc_id}'")
        return []

    logger.info(f"Generating summaries for {len(nodes)} nodes in doc '{tree.doc_id}'")
    summaries: list[NodeSummary] = []

    for node in nodes:
        # Skip nodes with very little text
        if len(node.text.strip()) < 50:
            logger.debug(f"Skipping node {node.id} ({node.label}): text too short")
            continue
        summary = generate_node_summary(node)
        summaries.append(summary)
        logger.info(f"Summary generated for {node.node_type} '{node.label}'")

    return summaries


def build_summary_context(summary: NodeSummary) -> str:
    """Build a rich text context from a pre-computed summary for use in RAG.

    This text is what gets sent to the LLM when a structural query matches
    a pre-computed summary instead of going through chunk retrieval.
    """
    parts: list[str] = []
    parts.append(f"=== {summary.label} ===")
    parts.append(f"Caminho: {summary.path}")

    if summary.resumo_executivo:
        parts.append(f"\nResumo executivo:\n{summary.resumo_executivo}")

    if summary.resumo_juridico:
        parts.append(f"\nResumo juridico:\n{summary.resumo_juridico}")

    if summary.pontos_chave:
        parts.append("\nPontos-chave:")
        for p in summary.pontos_chave:
            parts.append(f"  - {p}")

    if summary.artigos_cobertos:
        parts.append(f"\nArtigos cobertos: {', '.join(summary.artigos_cobertos)}")

    if summary.obrigacoes:
        parts.append("\nObrigacoes:")
        for o in summary.obrigacoes:
            parts.append(f"  - {o}")

    if summary.restricoes:
        parts.append("\nRestricoes/vedacoes:")
        for r in summary.restricoes:
            parts.append(f"  - {r}")

    if summary.definicoes:
        parts.append("\nDefinicoes:")
        for d in summary.definicoes:
            parts.append(f"  - {d}")

    return "\n".join(parts)
