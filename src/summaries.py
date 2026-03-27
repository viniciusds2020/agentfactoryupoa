"""Pre-computed summaries for legal document structural nodes.

Generates and stores summaries for chapters, sections, and titles via LLM.
Summaries include: executive summary, legal summary, key points, covered articles,
obligations, restrictions, and definitions.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

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


class SummaryStatus:
    GENERATED = "generated"
    INVALID = "invalid"
    FALLBACK_ONLY = "fallback_only"
    CACHED = "cached"


class _SummaryPayload(BaseModel):
    resumo_executivo: str = ""
    resumo_juridico: str = ""
    pontos_chave: list[str] = []
    artigos_cobertos: list[str] = []
    obrigacoes: list[str] = []
    restricoes: list[str] = []
    definicoes: list[str] = []


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
    source_hash: str = ""
    source_text_length: int = 0
    status: str = SummaryStatus.GENERATED
    validation_errors: list[str] = field(default_factory=list)
    generation_meta: dict[str, Any] = field(default_factory=dict)

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
            "source_hash": self.source_hash,
            "source_text_length": self.source_text_length,
            "status": self.status,
            "validation_errors": self.validation_errors,
            "generation_meta": self.generation_meta,
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
            source_hash=data.get("source_hash", ""),
            source_text_length=data.get("source_text_length", data.get("text_length", 0)),
            status=data.get("status", SummaryStatus.GENERATED),
            validation_errors=data.get("validation_errors", []),
            generation_meta=data.get("generation_meta", {}),
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


def _validate_summary_payload(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize summary payload from LLM."""
    errors: list[str] = []
    try:
        parsed = _SummaryPayload(**data)
        normalized = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed.dict()
    except ValidationError as exc:
        errors.extend(err.get("msg", "validation_error") for err in exc.errors())
        normalized = {}

    for key in (
        "pontos_chave",
        "artigos_cobertos",
        "obrigacoes",
        "restricoes",
        "definicoes",
    ):
        value = normalized.get(key, [])
        if not isinstance(value, list):
            errors.append(f"{key}: expected list")
            normalized[key] = []
            continue
        normalized[key] = [str(item).strip() for item in value if str(item).strip()]

    resumo_exec = str(normalized.get("resumo_executivo", "")).strip()
    resumo_jur = str(normalized.get("resumo_juridico", "")).strip()
    pontos = normalized.get("pontos_chave", [])
    if not resumo_exec and not resumo_jur and not pontos:
        errors.append("missing essential content (resumo_executivo/resumo_juridico/pontos_chave)")

    return normalized, errors


def _split_text_for_summary(text: str, max_chars: int = _MAX_SUMMARY_INPUT) -> list[str]:
    """Split long text into paragraph-aware chunks for map-reduce summarization."""
    clean = text.strip()
    if len(clean) <= max_chars:
        return [clean]
    blocks = [b.strip() for b in clean.split("\n\n") if b.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        if current and current_len + len(block) + 2 > max_chars:
            parts.append("\n\n".join(current))
            current = [block]
            current_len = len(block)
            continue
        current.append(block)
        current_len += len(block) + 2
    if current:
        parts.append("\n\n".join(current))
    return parts or [_truncate_text(clean, max_chars=max_chars)]


def _call_summary_llm(label: str, path: str, articles: str, text: str) -> dict[str, Any]:
    from src import llm
    user_msg = _SUMMARY_USER_TEMPLATE.format(
        label=label,
        path=path,
        articles=articles,
        text=text,
    )
    response = llm.chat(
        [{"role": "user", "content": user_msg}],
        system=_SUMMARY_SYSTEM_PROMPT,
    )
    return _parse_llm_json(response)


def _merge_partial_payloads(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {}
    resumo_exec = " ".join(str(p.get("resumo_executivo", "")).strip() for p in parts if str(p.get("resumo_executivo", "")).strip())
    resumo_jur = " ".join(str(p.get("resumo_juridico", "")).strip() for p in parts if str(p.get("resumo_juridico", "")).strip())

    def _merge_list(key: str, limit: int = 20) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for p in parts:
            for item in p.get(key, []) or []:
                val = str(item).strip()
                if not val:
                    continue
                norm = val.lower()
                if norm in seen:
                    continue
                seen.add(norm)
                out.append(val)
                if len(out) >= limit:
                    return out
        return out

    return {
        "resumo_executivo": resumo_exec.strip(),
        "resumo_juridico": resumo_jur.strip(),
        "pontos_chave": _merge_list("pontos_chave"),
        "artigos_cobertos": _merge_list("artigos_cobertos"),
        "obrigacoes": _merge_list("obrigacoes"),
        "restricoes": _merge_list("restricoes"),
        "definicoes": _merge_list("definicoes"),
    }


def _extractive_summary_payload(node: LegalNode) -> dict[str, Any]:
    """Deterministic fallback when LLM output is unavailable or malformed."""
    text = " ".join(line.strip() for line in (node.text or "").splitlines() if line.strip())
    sentences = [s.strip() for s in re.split(r"(?<=[\.;:])\s+", text) if s.strip()]
    lead = sentences[:5]
    key_points = [s[:220] for s in lead[:5]]
    executivo = " ".join(lead[:2]).strip() or text[:320].strip()
    juridico = " ".join(lead[:4]).strip() or executivo
    return {
        "resumo_executivo": executivo,
        "resumo_juridico": juridico,
        "pontos_chave": key_points,
        "artigos_cobertos": [f"Art. {a}" if not str(a).startswith("Art.") else str(a) for a in (node.articles or [])[:20]],
        "obrigacoes": [],
        "restricoes": [],
        "definicoes": [],
    }


def generate_node_summary(node: LegalNode) -> NodeSummary:
    """Generate a pre-computed summary for a single legal tree node via LLM.

    Uses lazy import of llm to avoid circular dependency.
    """
    full_text = node.text or ""
    text_parts = _split_text_for_summary(full_text, max_chars=_MAX_SUMMARY_INPUT)
    articles_str = ", ".join(node.articles[:50]) if node.articles else "Nenhum"
    source_hash = hashlib.sha256(full_text.encode("utf-8", errors="ignore")).hexdigest()

    payload: dict[str, Any] = {}
    validation_errors: list[str] = []
    generation_meta: dict[str, Any] = {
        "input_parts": len(text_parts),
        "input_chars": len(full_text),
        "truncated_single_pass": len(text_parts) == 1 and len(full_text) > _MAX_SUMMARY_INPUT,
    }

    try:
        if len(text_parts) == 1:
            payload = _call_summary_llm(
                label=node.label,
                path=node.path,
                articles=articles_str,
                text=_truncate_text(text_parts[0]),
            )
        else:
            partials: list[dict[str, Any]] = []
            for idx, part in enumerate(text_parts, start=1):
                part_payload = _call_summary_llm(
                    label=f"{node.label} (parte {idx}/{len(text_parts)})",
                    path=node.path,
                    articles=articles_str,
                    text=part,
                )
                partials.append(part_payload)
            payload = _merge_partial_payloads(partials)
            generation_meta["multi_part"] = True
            generation_meta["partial_summaries"] = len(partials)
        normalized, validation_errors = _validate_summary_payload(payload)
    except Exception as exc:
        logger.error(f"Failed to generate summary for node {node.id}: {exc}")
        normalized = {}
        validation_errors = [f"generation_error: {exc}"]

    status = SummaryStatus.GENERATED if not validation_errors else SummaryStatus.INVALID
    if status == SummaryStatus.INVALID:
        normalized = _extractive_summary_payload(node)
        fallback_errors = list(validation_errors)
        validation_errors = fallback_errors
        status = SummaryStatus.FALLBACK_ONLY

    return NodeSummary(
        node_id=node.id,
        node_type=node.node_type,
        label=node.label,
        path=node.path,
        resumo_executivo=normalized.get("resumo_executivo", ""),
        resumo_juridico=normalized.get("resumo_juridico", ""),
        pontos_chave=normalized.get("pontos_chave", []),
        artigos_cobertos=normalized.get("artigos_cobertos", []),
        obrigacoes=normalized.get("obrigacoes", []),
        restricoes=normalized.get("restricoes", []),
        definicoes=normalized.get("definicoes", []),
        text_length=len(full_text),
        source_hash=source_hash,
        source_text_length=len(full_text),
        status=status,
        validation_errors=validation_errors,
        generation_meta=generation_meta,
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


def generate_summary_from_scope_text(
    *,
    node_id: str,
    node_type: str,
    label: str,
    path: str,
    text: str,
    articles: list[str] | None = None,
    status_override: str = SummaryStatus.FALLBACK_ONLY,
) -> NodeSummary:
    """Generate summary for dynamically resolved structural scope (fallback path)."""
    synthetic = LegalNode(
        id=node_id,
        node_type=node_type,
        label=label,
        numeral="",
        text=text,
        path=path or label,
        articles=articles or [],
    )
    summary = generate_node_summary(synthetic)
    summary.status = status_override if summary.status != SummaryStatus.INVALID else SummaryStatus.INVALID
    return summary


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
