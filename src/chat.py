"""Semantic retrieval (vector search) and RAG generation."""
from __future__ import annotations

import re
import unicodedata
import json
from pathlib import Path
from dataclasses import dataclass, field

from src import llm, vectordb
from src.chat_intent import (
    QueryIntent,
    _classify_intent_regex,
    _cosine_similarity,
)
from src.chat_structural import (
    _boost_section_hint_compatibility,
    _enforce_summary_structural_scope,
    _expand_adjacent_structural_context,
    _expand_exact_chapter_context,
    _expand_refs_arabic_roman,
    _extract_article_refs,
    _extract_chapter_refs,
    _extract_chapter_refs_from_text,
    _extract_node_refs_from_text,
    _extract_refs_from_meta_value,
    _extract_structural_targets,
    _is_structural_query,
    _item_matches_chapter_refs,
    _item_matches_structural_target,
    _keyword_tokens,
    _metadata_hint_text,
    _metadata_structural_scope_from_seed,
    _metadata_structural_text,
    _normalize_for_match,
    _prioritize_exact_chapter_matches,
    _rerank_structural_continuity,
    _structural_match_strength,
    _supplement_chapter_matches,
)
from src.config import get_settings
from src.guardrails import detect_injection, sanitize_context_chunk, sanitize_history, sanitize_question
from src.lexical import normalize_query_numerals
from src.observability import metrics
from src.prompts import (
    build_catalog_record_messages,
    build_rag_messages,
    format_context,
    get_catalog_record_system,
    get_rag_system,
)
from src.table_renderer import render_table_answer
from src.table_semantics import aggregation_lead_text, infer_subject_label, render_value_by_unit
from src.utils import get_logger, log_event

logger = get_logger(__name__)


def classify_query_intent(question: str) -> QueryIntent:
    settings = get_settings()
    if getattr(settings, "intent_classifier", "regex") == "embeddings":
        return _classify_intent_embeddings(question)
    return _classify_intent_regex(question)


# ── Context budget management ────────────────────────────────────────────────

def _estimate_tokens(text: str, chars_per_token: float = 3.5) -> int:
    """Estimate token count from character length (conservative for PT-BR)."""
    return int(len(text) / chars_per_token)


def _trim_to_budget(
    items: list[dict],
    max_tokens: int,
    chars_per_token: float = 3.5,
) -> list[dict]:
    """Drop lowest-scoring items until total context fits within token budget.

    Preserves order of remaining items. Never removes the first item.
    """
    if not items or max_tokens <= 0:
        return items

    total = sum(_estimate_tokens(item.get("text", ""), chars_per_token) for item in items)
    if total <= max_tokens:
        return items

    scored = [(i, item, item.get("score", 0.0)) for i, item in enumerate(items)]
    droppable = sorted(scored[1:], key=lambda x: x[2])

    kept_indices = set(range(len(items)))
    current_total = total

    for idx, item, _score in droppable:
        if current_total <= max_tokens:
            break
        item_tokens = _estimate_tokens(item.get("text", ""), chars_per_token)
        kept_indices.discard(idx)
        current_total -= item_tokens

    result = [items[i] for i in sorted(kept_indices)]
    if len(result) < len(items):
        metrics.increment("chat.context_budget.trimmed")
        metrics.observe("chat.context_budget.dropped_chunks", len(items) - len(result))
    return result


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class Source:
    chunk_id: str
    doc_id: str
    excerpt: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ChatResult:
    answer: str
    sources: list[Source]
    request_id: str


def _filter_by_threshold(results: list[dict], min_score: float) -> list[dict]:
    """Remove results below minimum score. Always keeps at least one result."""
    if min_score <= 0 or not results:
        return results
    filtered = [r for r in results if r["score"] >= min_score]
    if not filtered:
        return results[:1]
    return filtered














_CHAPTER_REF_RE = re.compile(r"\bcapitulo\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_TITLE_REF_RE = re.compile(r"\btitulo\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"\bsecao\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_SUBSECTION_REF_RE = re.compile(r"\bsubsecao\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_ARTICLE_REF_RE = re.compile(
    r"\bart(?:igo)?\.?\s*([0-9]{1,4})(?:\s*(?:º|°|o|Âº|Â°|ª))?",
    re.IGNORECASE,
)
_ARTICLE_BLOCK_RE = re.compile(
    r"\bArt\.?\s*(\d{1,4})(?:\s*(?:º|°|o|Âº|Â°|ª))?",
    re.IGNORECASE,
)

_NODE_META_KEYS: dict[str, tuple[str, ...]] = {
    "titulo": ("titulo", "title"),
    "capitulo": ("capitulo", "section"),
    "secao": ("secao", "subsection"),
    "subsecao": ("subsection",),
    "artigo": ("artigo",),
}
_STRUCTURAL_LABEL_REF_RE: dict[str, re.Pattern] = {
    "titulo": re.compile(r"\btitulo\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE),
    "capitulo": re.compile(r"\bcapitulo\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE),
    "secao": re.compile(r"\bsecao\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE),
    "subsecao": re.compile(r"\bsubsecao\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE),
}



























































def _build_source_excerpt(item: dict, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", item.get("text", "")).strip()
    excerpt = text[:limit].strip()
    if len(text) > limit:
        excerpt += "..."
    return excerpt


def _mark_possible_contradiction(answer_text: str, results: list[dict], question: str) -> str:
    normalized_answer = _normalize_for_match(answer_text)
    if "nao encontrei" not in normalized_answer and "nao localizei" not in normalized_answer:
        return answer_text

    strong_match = next((item for item in results if _structural_match_strength(question, item) > 0), None)
    if not strong_match:
        return answer_text

    meta = strong_match.get("metadata", {})
    label = _metadata_structural_text(meta) or meta.get("doc_id", "trecho relevante")
    note = (
        f"\n\nObservacao: a busca localizou uma secao potencialmente relevante ({label}), "
        "mas o contexto recuperado parece parcial. Trate esta resposta com cautela e revise as fontes citadas."
    )
    if note.strip() in answer_text:
        return answer_text
    return answer_text + note


# ── Legal parent expansion ───────────────────────────────────────────────────

def _expand_legal_context(
    results: list[dict],
    physical_collection: str,
    max_parents: int = 5,
    max_expansion_tokens: int = 3000,
    chars_per_token: float = 3.5,
) -> list[dict]:
    """For child chunks, fetch their parent chunk to give the LLM full article context."""
    seen_ids = {r["id"] for r in results}
    parent_keys_seen: set[str] = set()
    parents_to_fetch: list[str] = []

    for r in results:
        meta = r.get("metadata", {})
        if meta.get("chunk_type") == "child":
            pk = meta.get("parent_key", "")
            if pk and pk not in parent_keys_seen:
                parent_keys_seen.add(pk)
                parents_to_fetch.append(pk)

    if not parents_to_fetch:
        return results

    expanded = list(results)
    expansion_tokens = 0

    for pk in parents_to_fetch[:max_parents]:
        items = vectordb.get_by_metadata(
            physical_collection,
            {"$and": [{"parent_key": {"$eq": pk}}, {"chunk_type": {"$eq": "parent"}}]},
        )
        for item in items:
            item_tokens = _estimate_tokens(item.get("text", ""), chars_per_token)
            if item["id"] not in seen_ids and expansion_tokens + item_tokens <= max_expansion_tokens:
                child_scores = [r.get("score", 0) for r in results if r.get("metadata", {}).get("parent_key") == pk]
                item["score"] = max(child_scores) if child_scores else 0.0
                expanded.append(item)
                seen_ids.add(item["id"])
                expansion_tokens += item_tokens

    return expanded


# ── Query expansion & HyDE ───────────────────────────────────────────────────

def _expand_query_llm(
    question: str,
    settings,
    request_id: str = "-",
) -> list[str]:
    """Generate LLM-based query reformulations. Returns [original, ...expansions]."""
    from src.prompts import build_query_expansion_messages

    n = getattr(settings, "query_expansion_max_reformulations", 2)
    messages = build_query_expansion_messages(question, n=n)
    try:
        response = llm.chat(messages, system="")
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        expansions = lines[:n]
        log_event(
            logger, 20, "Query expansion via LLM",
            original=question,
            expansions=expansions,
            request_id=request_id,
        )
    except Exception:
        log_event(logger, 30, "Query expansion failed, using original only", request_id=request_id)
        expansions = []

    return [question] + expansions


def _generate_hypothetical_doc(
    question: str,
    settings,
    request_id: str = "-",
) -> str:
    """Generate a hypothetical answer for HyDE embedding."""
    from src.prompts import build_hyde_messages

    messages = build_hyde_messages(question)
    try:
        hypo = llm.chat(messages, system="")
        log_event(
            logger, 20, "HyDE document generated",
            question=question,
            hypo_length=len(hypo),
            request_id=request_id,
        )
        return hypo.strip()
    except Exception:
        log_event(logger, 30, "HyDE generation failed, using original query", request_id=request_id)
        return question


# ── Semantic retrieval ────────────────────────────────────────────────────────

def _deduplicate_hits(hits: list[dict]) -> list[dict]:
    """Deduplicate vector results by ID, keeping the highest-scored entry."""
    seen: dict[str, dict] = {}
    for item in hits:
        cid = item["id"]
        if cid not in seen or item.get("score", 0) > seen[cid].get("score", 0):
            seen[cid] = item
    return sorted(seen.values(), key=lambda x: x.get("score", 0), reverse=True)


def _vector_retrieve(
    question: str,
    physical_collection: str,
    top_k: int,
    settings,
    where: dict | None = None,
    model_name: str = "",
    request_id: str = "-",
) -> list[dict]:
    """Run semantic (vector) retrieval with optional enhancements.

    Optional enhancements (controlled via settings):
    - query_expansion_enabled: multi-query retrieval via LLM reformulations
    - hyde_enabled: embed a hypothetical answer for vector search
    - reranker_enabled: cross-encoder reranking after initial retrieval
    """
    fetch_k = top_k * 3

    # ── Determine queries to run ─────────────────────────────────────────
    queries = [question]
    if getattr(settings, "query_expansion_enabled", False):
        with metrics.time_block("chat.query_expansion"):
            queries = _expand_query_llm(question, settings, request_id)

    search_query = normalize_query_numerals(question)

    # ── Vector search ────────────────────────────────────────────────────
    all_hits: list[dict] = []
    if getattr(settings, "hyde_enabled", False):
        with metrics.time_block("chat.hyde_generation"):
            hypo_doc = _generate_hypothetical_doc(question, settings, request_id)
        with metrics.time_block("chat.embed_hyde"):
            hyde_vec = llm.embed([hypo_doc], model_name=model_name)[0]
        with metrics.time_block("chat.vector_query_hyde"):
            all_hits.extend(
                vectordb.query(physical_collection, [hyde_vec], n_results=fetch_k, where=where)
            )
        if getattr(settings, "hyde_merge_original", True):
            with metrics.time_block("chat.embed_query"):
                orig_vec = llm.embed([search_query], model_name=model_name)[0]
            with metrics.time_block("chat.vector_query_original"):
                all_hits.extend(
                    vectordb.query(physical_collection, [orig_vec], n_results=fetch_k, where=where)
                )
    else:
        for q in queries:
            q_normalized = normalize_query_numerals(q)
            with metrics.time_block("chat.embed_query"):
                q_vec = llm.embed([q_normalized], model_name=model_name)[0]
            with metrics.time_block("chat.vector_query"):
                all_hits.extend(
                    vectordb.query(physical_collection, [q_vec], n_results=fetch_k, where=where)
                )

    # ── Convert distance → score (cosine distance 0-2 → score 0-1) ─────
    for hit in all_hits:
        if "score" not in hit and "distance" in hit:
            hit["score"] = max(0.0, 1.0 - hit["distance"])

    # ── Deduplicate ──────────────────────────────────────────────────────
    results = _deduplicate_hits(all_hits)

    log_event(
        logger, 20, "Retrieval completed",
        collection=physical_collection,
        vector_hits=len(all_hits),
        unique_results=len(results),
        fetch_k=fetch_k,
        num_queries=len(queries),
        request_id=request_id,
    )

    # ── Cross-encoder reranking ──────────────────────────────────────────
    if getattr(settings, "reranker_enabled", False):
        from src.reranker import rerank as ce_rerank

        reranker_top = getattr(settings, "reranker_top_k", top_k)
        # For structural legal queries, avoid truncating candidates too early.
        # We keep a larger candidate set so chapter/section boosts can act after reranking.
        if _is_structural_query(question):
            reranker_top = max(reranker_top, min(len(results), fetch_k))
        else:
            reranker_top = max(reranker_top, top_k)
        results = ce_rerank(
            query=question,
            candidates=results,
            model_name=settings.reranker_model,
            top_k=reranker_top,
        )

    # ── Structural reranking + threshold ─────────────────────────────────
    results = _rerank_structural_continuity(
        results,
        question=question,
        bonus=getattr(settings, "retrieval_structural_bonus", 0.0),
    )[:top_k]

    return _filter_by_threshold(results, settings.retrieval_min_score)


def _structured_rows_to_results(rows: list[dict], top_k: int) -> list[dict]:
    results: list[dict] = []
    for i, row in enumerate(rows[:top_k]):
        doc_id = str(row.get("doc_id", ""))
        row_index = row.get("row_index", i)
        canon = str(row.get("texto_canonico", "")).strip()
        if not canon:
            continue
        page = row.get("page_number")
        metadata = {
            "doc_id": doc_id,
            "chunk_type": "tabular_structured",
            "row_index": row_index,
            "source": "structured_store",
        }
        if page is not None:
            metadata["page_number"] = page
        results.append(
            {
                "id": f"structured::{doc_id}::{row_index}",
                "text": canon,
                "metadata": metadata,
                "score": max(0.5, 1.0 - (i * 0.03)),
            }
        )
    return results


def _merge_structured_vector(structured: list[dict], vector: list[dict], top_k: int) -> list[dict]:
    merged: dict[str, dict] = {}
    for item in structured + vector:
        cid = item["id"]
        if cid not in merged or item.get("score", 0.0) > merged[cid].get("score", 0.0):
            merged[cid] = item
    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked[:top_k]


def _format_numeric_value(value) -> str:
    if value is None:
        return "sem resultado"
    try:
        num = float(value)
    except Exception:
        return str(value)
    return f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _format_count_value(value) -> str:
    if value is None:
        return "0"
    try:
        return str(int(round(float(value))))
    except Exception:
        return str(value)


def _humanize_identifier(value: str) -> str:
    return str(value or "").replace("_", " ").strip()


def _humanize_plural_identifier(value: str) -> str:
    text = _humanize_identifier(value)
    if not text:
        return text
    return text if text.endswith("s") else f"{text}s"


def _get_collection_context_hint(workspace_id: str, collection: str) -> str:
    from src import controlplane

    try:
        return controlplane.get_collection_context(workspace_id, collection)
    except Exception:
        return ""


def _build_table_query_summary(plan: dict) -> str:
    sql_generated = str(plan.get("sql_generated") or "").strip()
    if sql_generated:
        return sql_generated
    op = str(plan.get("operation", ""))
    agg = str(plan.get("aggregation", "")).upper()
    metric = str(plan.get("metric_column") or "*")
    filters = plan.get("filters", []) or []
    group_by = plan.get("group_by", []) or []
    where_clause = " AND ".join(f"{flt['column']} = '{flt['value']}'" for flt in filters) if filters else "1=1"
    if op == "aggregate":
        expr = "COUNT(*)" if agg == "COUNT" or metric == "*" else f"{agg}({metric})"
        return f"SELECT {expr} FROM tabela WHERE {where_clause}"
    if op == "groupby":
        groups = ", ".join(group_by)
        expr = "COUNT(*)" if agg == "COUNT" or metric == "*" else f"{agg}({metric})"
        return f"SELECT {groups}, {expr} FROM tabela WHERE {where_clause} GROUP BY {groups} ORDER BY 2 DESC"
    if op == "rank":
        return f"SELECT ... FROM tabela WHERE {where_clause} ORDER BY {metric} DESC LIMIT {int(plan.get('limit') or 10)}"
    if op == "distinct":
        dimension = str(plan.get("dimension_column") or "*")
        return f"SELECT DISTINCT {dimension} FROM tabela WHERE {where_clause} ORDER BY {dimension} ASC"
    if op == "schema":
        return "SHOW COLUMNS FROM tabela"
    if op == "describe_column":
        return f"DESCRIBE COLUMN {str(plan.get('target_column') or '').strip()}"
    if op == "catalog_lookup":
        return "LOOKUP registro por identificador ou descricao"
    if op == "catalog_field_lookup":
        return f"LOOKUP campo {str(plan.get('target_column') or '').strip()} em registro de catalogo"
    if op == "catalog_record_summary":
        return "SUMMARY de registro de catalogo"
    if op == "catalog_compare":
        return "COMPARE registros de catalogo"
    return "Consulta analitica estruturada"


def _describe_filters(filters: list[dict]) -> str:
    if not filters:
        return ""
    parts = []
    for flt in filters:
        col = _humanize_identifier(flt.get("column", ""))
        val = str(flt.get("value", "")).strip()
        parts.append(f"{col} = {val}")
    return ", ".join(parts)


def _business_filter_phrase(filters: list[dict]) -> str:
    if not filters:
        return ""
    parts = []
    for flt in filters:
        col = _humanize_identifier(flt.get("column", ""))
        val = str(flt.get("value", "")).strip()
        if col == "estado":
            parts.append(f"no estado de {val}")
        elif col == "cidade":
            parts.append(f"na cidade de {val}")
        else:
            parts.append(f"com {col} = {val}")
    return " e ".join(parts)


def _table_result_preview(result: dict, plan: dict) -> str:
    if result.get("operation") == "aggregate":
        metric = _humanize_identifier(plan.get("metric_column") or "registros")
        unit = str(plan.get("metric_unit") or ("count" if plan.get("aggregation") == "count" else "number"))
        formatter = _format_count_value if plan.get("aggregation") == "count" else lambda value: render_value_by_unit(value, unit)
        return f"resultado: {formatter(result.get('value'))}; metrica: {metric}; agregacao: {plan.get('aggregation', '')}"
    if result.get("operation") == "distinct":
        values = result.get("values", [])[:10]
        return f"valores distintos ({result.get('count', len(values))}): " + ", ".join(str(v) for v in values)
    if result.get("operation") == "schema":
        cols = result.get("columns", [])[:20]
        return f"colunas ({result.get('count', len(cols))}): " + ", ".join(str(v) for v in cols)
    if result.get("operation") == "describe_column":
        target = result.get("target_profile") or {}
        if not target:
            return "coluna nao encontrada"
        return (
            f"coluna: {target.get('name', '')}; tipo_semantico: {target.get('semantic_type', '')}; "
            f"unidade: {target.get('unit', '')}; descricao: {target.get('description', '')}"
        )
    if result.get("operation") == "compare":
        rows = result.get("rows", [])[:5]
        return "; ".join(
            ", ".join(f"{key}={value}" for key, value in row.items())
            for row in rows
        ) or "resultado: sem comparacao"
    if result.get("operation") in {"catalog_lookup", "catalog_field_lookup", "catalog_record_summary"}:
        record = result.get("record") or {}
        if not record:
            return "registro nao encontrado"
        interesting = [
            "procedimento",
            "codigo",
            "descricao_unimed_poa",
            "descricao",
            "cobertura_unimed_poa",
            "cobertura",
            "prazo_autorizacao_conforme_rn_n_623_ans",
            "orientacao_autorizacao_call_center",
        ]
        preview_parts = [f"{key}={record.get(key)}" for key in interesting if key in record and str(record.get(key, "")).strip()]
        return "; ".join(preview_parts[:6]) or "registro localizado"
    if result.get("operation") == "catalog_compare":
        rows = result.get("rows", [])[:3]
        return "; ".join(
            ", ".join(f"{key}={value}" for key, value in row.items() if str(value).strip())
            for row in rows
        ) or "comparacao de catalogo sem resultado"
    rows = result.get("rows", [])[:5]
    if not rows:
        return "resultado: sem linhas"
    cols = [c for c in rows[0].keys()]
    preview_lines = []
    for row in rows:
        preview_lines.append(" | ".join(f"{col}={row.get(col)}" for col in cols))
    return "\n".join(preview_lines)


def _answer_table_first(
    *,
    collection: str,
    question: str,
    request_id: str,
    workspace_id: str = "default",
) -> ChatResult | None:
    from src.structured_store import execute_plan, has_structured_data, plan_query

    settings = get_settings()
    if not (getattr(settings, "query_routing_enabled", False) and getattr(settings, "structured_store_enabled", False)):
        return None
    if not has_structured_data(collection):
        return None

    context_hint = _get_collection_context_hint(workspace_id, collection)
    plan = plan_query(collection, question, context_hint=context_hint)
    if not plan:
        return None
    result = execute_plan(collection, plan)
    if not result:
        return None
    if result.get("sql_generated"):
        plan["sql_generated"] = result.get("sql_generated", "")

    filters = plan.get("filters", [])
    business_scope = _business_filter_phrase(filters)
    answer, excerpt = render_table_answer(
        question=question,
        plan=plan,
        result=result,
        context_hint=context_hint,
        business_scope=business_scope,
    )

    try:
        from src import controlplane

        controlplane.log_query_plan(
            workspace_id=workspace_id,
            collection=collection,
            question=question,
            planner_source=str(plan.get("planner_source") or "heuristic"),
            plan=plan,
            validated=bool(plan.get("validated", False)),
            validation_errors=list(plan.get("validation_errors", []) or []),
            sql_generated=str(result.get("sql_generated") or ""),
        )
    except Exception:
        pass

    log_event(
        logger,
        20,
        "Table-first query answered",
        request_id=request_id,
        collection=collection,
        operation=plan.get("operation", ""),
        aggregation=plan.get("aggregation", ""),
        metric=plan.get("metric_column", ""),
        filters=filters,
        planner_source=plan.get("planner_source", ""),
    )
    metrics.increment("chat.table_first.hit")
    if plan.get("validated", False):
        metrics.increment("chat.table_first.validated")
    operation = str(plan.get("operation") or "")
    citation_label = "consulta analitica executada"
    if operation.startswith("catalog_"):
        citation_label = "registro de catalogo"
    return ChatResult(
        answer=answer,
        sources=[
            Source(
                chunk_id=f"structured::{collection}::table_first",
                doc_id="",
                excerpt=excerpt[:400],
                score=1.0,
                metadata={
                    "source": "structured_store",
                    "source_kind": "table_query",
                    "plan": plan,
                    "result": result,
                    "query_summary": _build_table_query_summary({**plan, "sql_generated": result.get("sql_generated", "")}),
                    "result_preview": _table_result_preview(result, plan),
                    "citation_label": citation_label,
                    "context_hint": context_hint,
                },
            )
        ],
        request_id=request_id,
    )


def _retrieve_with_routing(
    question: str,
    collection: str,
    physical_collection: str,
    settings,
    top_k: int,
    where: dict | None = None,
    model_name: str = "",
    request_id: str = "-",
) -> list[dict]:
    if not (getattr(settings, "query_routing_enabled", False) and getattr(settings, "structured_store_enabled", False)):
        return _vector_retrieve(
            question=question,
            physical_collection=physical_collection,
            top_k=top_k,
            settings=settings,
            where=where,
            model_name=model_name,
            request_id=request_id,
        )

    from src.query_router import detect_query_intent
    from src.structured_store import has_structured_data, query_structured

    has_structured = has_structured_data(collection)
    intent = detect_query_intent(question, has_structured)
    log_event(
        logger,
        20,
        "Query routed",
        route=intent.route,
        filters=intent.structured_filters,
        confidence=intent.confidence,
        collection=collection,
        request_id=request_id,
    )

    if intent.route == "vector":
        return _vector_retrieve(
            question=question,
            physical_collection=physical_collection,
            top_k=top_k,
            settings=settings,
            where=where,
            model_name=model_name,
            request_id=request_id,
        )

    structured_rows = query_structured(collection, intent.structured_filters, limit=top_k * 3)
    structured_results = _structured_rows_to_results(structured_rows, top_k=top_k * 3)

    if intent.route == "structured":
        if structured_results:
            return structured_results[:top_k]
        return _vector_retrieve(
            question=question,
            physical_collection=physical_collection,
            top_k=top_k,
            settings=settings,
            where=where,
            model_name=model_name,
            request_id=request_id,
        )

    vector_results = _vector_retrieve(
        question=question,
        physical_collection=physical_collection,
        top_k=top_k,
        settings=settings,
        where=where,
        model_name=model_name,
        request_id=request_id,
    )
    if not structured_results:
        return vector_results
    return _merge_structured_vector(structured_results, vector_results, top_k=top_k)


# ── Query intent classification (Gap 4) ─────────────────────────────────────












# ── Embeddings-based intent classifier ─────────────────────────────────────

# Exemplar queries per intent — used as few-shot anchors for similarity
_INTENT_EXEMPLARS: dict[QueryIntent, list[str]] = {
    QueryIntent.COUNT_STRUCTURAL: [
        "Quantos capitulos tem no estatuto?",
        "Quantas secoes existem no capitulo V?",
        "Numero de artigos do capitulo II",
    ],
    QueryIntent.LIST_STRUCTURAL: [
        "Quais capitulos existem no estatuto?",
        "Liste as secoes do capitulo V",
        "Quais artigos estao no capitulo II?",
    ],
    QueryIntent.LOCATE_STRUCTURAL: [
        "Qual e o ultimo capitulo?",
        "Qual e a primeira secao do capitulo V?",
        "Onde esta o capitulo VII?",
    ],
    QueryIntent.CONTAINS_STRUCTURAL: [
        "O capitulo V tem secao III?",
        "Existe art. 41 no capitulo X?",
        "O titulo I contem capitulo II?",
    ],
    QueryIntent.SUMMARY_STRUCTURAL: [
        "Resuma o capítulo II",
        "Sintetize a seção III",
        "Visão geral do título I",
        "Explique o capítulo 3 em linguagem simples",
        "Faça um resumo executivo do capítulo IV",
    ],
    QueryIntent.QUESTION_STRUCTURAL: [
        "O que diz o capítulo II sobre obrigações?",
        "Quais são os direitos previstos na seção I?",
        "O capítulo 5 trata de quê?",
        "Que artigos estão no título III?",
    ],
    QueryIntent.QUESTION_FACTUAL: [
        "Qual o prazo para pagamento de rescisão?",
        "É proibido remunerar diretores?",
        "Quantos membros tem o conselho?",
        "Qual é o quórum de deliberação?",
    ],
    QueryIntent.LOCATE_EXCERPT: [
        "Art. 41",
        "Artigo 15",
        "Capítulo II",
        "Seção III",
    ],
    QueryIntent.COMPARISON: [
        "Compare o capítulo 1 e o capítulo 2",
        "Qual a diferença entre a seção I e a seção II?",
        "Capítulo III versus capítulo IV",
    ],
}

# Cache for pre-computed exemplar embeddings (populated on first call)
_intent_exemplar_embeddings: dict[QueryIntent, list[list[float]]] | None = None


def _get_intent_exemplar_embeddings() -> dict[QueryIntent, list[list[float]]]:
    """Compute and cache exemplar embeddings using local chat module state.

    We keep this wrapper in `src.chat` for test compatibility: several tests
    monkeypatch `_intent_exemplar_embeddings`, `_INTENT_EXEMPLARS`, and `llm`
    directly on this module.
    """
    global _intent_exemplar_embeddings
    if _intent_exemplar_embeddings is not None:
        return _intent_exemplar_embeddings

    _intent_exemplar_embeddings = {}
    for intent, exemplars in _INTENT_EXEMPLARS.items():
        _intent_exemplar_embeddings[intent] = llm.embed(exemplars)

    log_event(
        logger,
        20,
        "Intent exemplar embeddings computed",
        intents=len(_intent_exemplar_embeddings),
    )
    return _intent_exemplar_embeddings


def _classify_intent_embeddings(question: str) -> QueryIntent:
    """Embedding-based intent classification backed by local module state."""
    exemplar_embs = _get_intent_exemplar_embeddings()
    q_embedding = llm.embed([question.strip()])[0]

    best_intent = QueryIntent.QUESTION_FACTUAL
    best_score = -1.0

    for intent, emb_list in exemplar_embs.items():
        scores = [_cosine_similarity(q_embedding, emb) for emb in emb_list]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_intent = intent

    log_event(
        logger,
        10,
        "Intent classified via embeddings",
        question=question[:80],
        intent=best_intent.value,
        score=round(best_score, 4),
    )
    return best_intent











def _resolve_structural_summary(
    question: str,
    collection: str,
    workspace_id: str,
) -> dict | None:
    """Resolve summary_structural query with structural-type awareness and quality gating."""
    from src import controlplane

    targets = _extract_structural_targets(question)
    if not targets:
        refs = _extract_chapter_refs(question)
        if refs:
            targets = [("capitulo", refs)]
    if not targets:
        return None

    for node_type, refs in targets:
        search_terms = set()
        for ref in refs:
            search_terms.add(ref)
            search_terms.add(ref.upper())
            if ref.isdigit():
                from src.lexical import int_to_roman

                search_terms.add(int_to_roman(int(ref)))
            elif ref.upper().isalpha():
                from src.lexical import roman_to_int

                arabic = roman_to_int(ref.upper())
                if arabic > 0:
                    search_terms.add(str(arabic))

        for term in search_terms:
            summaries = controlplane.find_summaries_by_label(
                workspace_id=workspace_id,
                collection=collection,
                label_query=term,
                node_type=node_type,
            )
            summaries = [s for s in summaries if _summary_is_usable(s)]
            if summaries:
                chosen = summaries[0]
                log_event(
                    logger,
                    20,
                    "Pre-computed structural summary resolved",
                    node_type=node_type,
                    label=chosen.get("label", ""),
                    status=chosen.get("status", ""),
                )
                metrics.increment("chat.summary.cache_hit")
                return chosen

    metrics.increment("chat.summary.cache_miss")
    return None


def _answer_from_summary(
    summary: dict,
    question: str,
    history: list[ChatMessage] | None,
    request_id: str,
    profile_name: str,
    collection: str,
    physical_collection: str,
    workspace_id: str,
    model_name: str,
) -> ChatResult:
    """Generate answer from a pre-computed summary + supporting chunks."""
    from src.summaries import NodeSummary, build_summary_context

    node_summary = NodeSummary.from_dict(summary)
    summary_text = build_summary_context(node_summary)

    # Also fetch supporting chunks from FAISS for citations
    settings = get_settings()
    supporting = _vector_retrieve(
        question=question,
        physical_collection=physical_collection,
        top_k=min(5, settings.retrieval_top_k),
        settings=settings,
        model_name=model_name,
        request_id=request_id,
    )
    # Prioritize chunks matching the chapter
    supporting = _prioritize_exact_chapter_matches(supporting, question)

    # Build context: summary first, then supporting chunks
    context_parts = [f"[1] {summary_text}"]
    for i, item in enumerate(supporting[:4]):
        text = sanitize_context_chunk(item.get("text", ""))
        meta = item.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        context_parts.append(f"[{i + 2}] (Fonte: {doc_id}) {text}")

    context = "\n\n".join(context_parts)
    messages: list[dict] = [{"role": msg.role, "content": msg.content} for msg in (history or [])]
    messages.extend(build_rag_messages(context=context, question=question))

    with metrics.time_block("chat.llm_generation"):
        answer_text = llm.chat(messages, system=get_rag_system(profile_name))

    sources = [
        Source(
            chunk_id=f"summary::{node_summary.node_id}",
            doc_id="",
            excerpt=node_summary.resumo_executivo[:240],
            score=1.0,
            metadata={"node_type": node_summary.node_type, "label": node_summary.label, "path": node_summary.path},
        )
    ]
    for item in supporting[:4]:
        sources.append(Source(
            chunk_id=item["id"],
            doc_id=item.get("metadata", {}).get("doc_id", ""),
            excerpt=_build_source_excerpt(item),
            score=item.get("score", 0.0),
            metadata=item.get("metadata", {}),
        ))

    log_event(
        logger, 20, "Chat answered via pre-computed summary",
        collection=collection, node_id=node_summary.node_id,
        label=node_summary.label, request_id=request_id,
        status=node_summary.status,
    )
    metrics.increment(f"chat.summary.answer_from_{node_summary.status or 'unknown'}")
    return ChatResult(answer=answer_text, sources=sources, request_id=request_id)


def _summary_is_usable(summary: dict | None) -> bool:
    if not summary:
        return False
    status = str(summary.get("status", "generated")).lower()
    if status == "invalid":
        return False
    if not (
        str(summary.get("resumo_executivo", "")).strip()
        or str(summary.get("resumo_juridico", "")).strip()
        or summary.get("pontos_chave")
    ):
        return False
    return True


def _artifact_candidates(doc_id: str, filename: str) -> list[str]:
    candidates = [doc_id, Path(doc_id).name, filename, Path(filename).name, Path(filename).stem]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in candidates:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _resolve_processed_json(doc_id: str, filename: str) -> Path | None:
    artifacts_dir = Path(get_settings().pdf_pipeline_artifacts_dir)
    for key in _artifact_candidates(doc_id, filename):
        path = artifacts_dir / f"{key}.json"
        if path.exists():
            return path
    return None


def _extract_catalog_code(question: str) -> str:
    match = re.search(r"\b(\d{5,})\b", question or "")
    return match.group(1) if match else ""


def _catalog_field_request(question: str, workspace_id: str, collection: str) -> tuple[str, str, str] | None:
    q_norm = _normalize_for_match(question)
    semantic_priority = ["deadline_rule", "authorization_rule", "coverage_rule", "flag_boolean", "catalog_title"]
    semantic_patterns = {
        "deadline_rule": [
            r"\bprazo\b",
            r"\bprazo\s+de\s+autoriz",
            r"\bprazo\s+autoriz",
            r"\btempo\b",
        ],
        "authorization_rule": [
            r"\bautoriz",
            r"\bliberac",
        ],
        "coverage_rule": [
            r"\bcobertura\b",
            r"\bcobre\b",
        ],
        "flag_boolean": [
            r"\bemerg[eê]ncia\b",
            r"\burg[eê]ncia\b",
        ],
        "catalog_title": [
            r"\bdescri",
            r"\bnome\b",
            r"\btitulo\b",
            r"\bprocedimento\b",
        ],
    }
    fallback_tokens = {
        "coverage_rule": ("cobertura", "cobertura", "cobertura"),
        "authorization_rule": ("autorizacao", "autorizacao", "autorizacao"),
        "deadline_rule": ("prazo", "prazo", "prazo de autorizacao"),
        "flag_boolean": ("emergencia", "emergencia", "emergencia"),
        "catalog_title": ("descricao", "descricao", "descricao do procedimento"),
    }
    try:
        from src import controlplane

        profiles = controlplane.list_column_profiles(workspace_id, collection)
    except Exception:
        profiles = []

    allowed_semantics = {"coverage_rule", "authorization_rule", "deadline_rule", "catalog_title", "flag_boolean"}
    candidates: list[tuple[int, int, str, str, str]] = []
    for profile in profiles:
        semantic_type = str(profile.get("semantic_type", "")).strip()
        if semantic_type not in allowed_semantics:
            continue
        aliases = [str(alias).strip().lower() for alias in profile.get("aliases_json", "").split("||") if str(alias).strip()]
        aliases.extend(str(alias).strip().lower() for alias in profile.get("aliases", []) if str(alias).strip())
        name = str(profile.get("column_name") or profile.get("name") or "").strip()
        if not name:
            continue
        score = 0
        for pattern in semantic_patterns.get(semantic_type, []):
            if re.search(pattern, q_norm):
                score += 10
        for alias in aliases:
            if alias and re.search(rf"\b{re.escape(alias)}\b", q_norm):
                score += max(3, min(len(alias.split()), 4))
        if score <= 0:
            continue
        label = aliases[0] if aliases else name.replace("_", " ")
        priority = semantic_priority.index(semantic_type)
        candidates.append((score, -priority, name, semantic_type, label))

    if candidates:
        candidates.sort(reverse=True)
        _, _, name, semantic_type, label = candidates[0]
        return name, semantic_type, label

    for semantic_type in semantic_priority:
        token, column_name, label = fallback_tokens[semantic_type]
        if re.search(rf"\b{re.escape(token)}\b", q_norm):
            return column_name, semantic_type, label
    return None


def _catalog_profiles(workspace_id: str, collection: str) -> list[dict]:
    try:
        from src import controlplane

        raw_profiles = controlplane.list_column_profiles(workspace_id, collection)
    except Exception:
        return []

    normalized: list[dict] = []
    for profile in raw_profiles:
        aliases = profile.get("aliases")
        if aliases is None:
            aliases_json = str(profile.get("aliases_json", "")).strip()
            if aliases_json.startswith("["):
                try:
                    aliases = json.loads(aliases_json)
                except Exception:
                    aliases = []
            else:
                aliases = []
        normalized.append(
            {
                "name": str(profile.get("column_name") or profile.get("name") or "").strip(),
                "display_name": str(profile.get("display_name") or profile.get("column_name") or profile.get("name") or "").strip(),
                "semantic_type": str(profile.get("semantic_type", "")).strip(),
                "role": str(profile.get("role", "")).strip(),
                "aliases": [str(alias).strip().lower() for alias in aliases or [] if str(alias).strip()],
            }
        )
    return [profile for profile in normalized if profile.get("name")]


def _catalog_header_order(blocks: list[dict], start_idx: int, profiles: list[dict]) -> list[dict]:
    if not profiles:
        return []
    search_start = max(0, start_idx - 120)
    best_matches: list[tuple[int, dict]] = []
    best_key: tuple[int, int] | None = None
    for idx in range(search_start, start_idx + 1):
        text_parts: list[str] = []
        consumed = 0
        probe = idx
        while probe <= start_idx and consumed < 5:
            raw = str(blocks[probe].get("text", "")).strip()
            if not raw:
                probe += 1
                consumed += 1
                continue
            if probe > idx and _looks_like_catalog_record_start(raw):
                break
            text_parts.append(raw.replace("\n", " "))
            probe += 1
            consumed += 1
        text = _normalize_for_match(" ".join(text_parts))
        local_matches: list[tuple[int, dict]] = []
        for profile in profiles:
            best_pos = None
            for alias in profile.get("aliases", []) or []:
                if not alias:
                    continue
                pos = text.find(_normalize_for_match(alias))
                if pos >= 0 and (best_pos is None or pos < best_pos):
                    best_pos = pos
            if best_pos is not None:
                local_matches.append((best_pos, profile))
        candidate_key = (len(local_matches), idx)
        if local_matches and (
            best_key is None
            or candidate_key[0] > best_key[0]
            or (candidate_key[0] == best_key[0] and candidate_key[1] > best_key[1])
        ):
            best_matches = sorted(local_matches, key=lambda item: item[0])
            best_key = candidate_key
    return [profile for _, profile in best_matches]


def _looks_like_catalog_record_start(text: str) -> bool:
    return bool(re.match(r"^\s*(?:[A-Z]{0,4}[-_/]?)?\d{5,}\b", text or ""))


def _split_non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in re.split(r"[\r\n]+", text or "") if line.strip()]


def _catalog_block_is_headerish(text: str, profiles: list[dict]) -> bool:
    normalized = _normalize_for_match(str(text or "").replace("\n", " "))
    if not normalized:
        return False
    hits = 0
    seen: set[str] = set()
    for profile in profiles:
        name = str(profile.get("name", "")).strip()
        for alias in profile.get("aliases", []) or []:
            alias_norm = _normalize_for_match(alias)
            if alias_norm and alias_norm in normalized:
                if name and name not in seen:
                    seen.add(name)
                    hits += 1
                break
    return hits >= 3


def _catalog_record_end_index(blocks: list[dict], start_idx: int) -> int:
    idx = start_idx + 1
    while idx < len(blocks):
        raw_text = str(blocks[idx].get("text", "")).strip()
        if raw_text and _looks_like_catalog_record_start(raw_text):
            return idx
        idx += 1
    return len(blocks)


def _extract_catalog_record_from_blocks(blocks: list[dict], start_idx: int, code: str, profiles: list[dict] | None = None) -> dict[str, str]:
    profiles = profiles or []
    identifier_profile = next((profile for profile in profiles if profile.get("role") == "identifier"), None)
    title_profile = next((profile for profile in profiles if profile.get("semantic_type") == "catalog_title"), None)
    auth_profile = next((profile for profile in profiles if profile.get("semantic_type") == "authorization_rule"), None)
    identifier_key = str(identifier_profile.get("name") if identifier_profile else "procedimento")
    title_key = str(title_profile.get("name") if title_profile else "descricao_unimed_poa")
    auth_key = str(
        auth_profile.get("name") if auth_profile else "orientacao_autorizacao_call_center"
    )
    record: dict[str, str] = {identifier_key: code}
    description_parts: list[str] = []
    field_lines: list[str] = []
    last_bucket = ""
    end_idx = _catalog_record_end_index(blocks, start_idx)
    start_text = str(blocks[start_idx].get("text", "")).strip()
    if start_text:
        trimmed_start = re.sub(rf"^\s*{re.escape(code)}\s*", "", start_text).strip(" -:/|")
        if trimmed_start:
            description_parts.append(trimmed_start)

    idx = start_idx + 1
    while idx < end_idx:
        raw_text = str(blocks[idx].get("text", "")).strip()
        if not raw_text:
            idx += 1
            continue
        if _catalog_block_is_headerish(raw_text, profiles):
            idx += 1
            continue

        lines = _split_non_empty_lines(raw_text)
        joined = " ".join(lines)
        lower_joined = _normalize_for_match(joined)

        if "autoriz" in lower_joined:
            field_lines.extend(lines)
            last_bucket = "auth"
        elif (
            len(lines) >= 3
            or any(token in lower_joined for token in ("sem cobertura", "hospitalar", "ambulatorial", "regulamentados", "nao", "sim", "uteis", "imediato"))
        ):
            field_lines.extend(lines)
            last_bucket = "field"
        elif last_bucket == "auth" and len(lines) <= 2:
            field_lines.extend(lines)
        else:
            description_parts.append(joined)
            last_bucket = "desc"
        idx += 1

    if description_parts:
        record[title_key] = re.sub(r"\s+", " ", " ".join(description_parts)).strip()

    normalized_lines = [re.sub(r"\s+", " ", line).strip() for line in field_lines if re.sub(r"\s+", " ", line).strip()]
    first_auth_idx = next(
        (i for i, line in enumerate(normalized_lines) if "autoriz" in _normalize_for_match(line)),
        -1,
    )
    if first_auth_idx >= 0:
        auth_lines = normalized_lines[first_auth_idx:]
        non_auth_lines = normalized_lines[:first_auth_idx]
    else:
        auth_lines = []
        non_auth_lines = normalized_lines

    if len(non_auth_lines) >= 5:
        first = _normalize_for_match(non_auth_lines[0])
        second = _normalize_for_match(non_auth_lines[1])
        coverage_tokens = ("hospitalar", "ambulatorial", "sem cobertura")
        emergency_tokens = {"nao", "n�o", "sim"}
        if (
            not any(token in first for token in coverage_tokens)
            and not any(token in second for token in coverage_tokens)
            and first not in emergency_tokens
            and second not in emergency_tokens
        ):
            non_auth_lines = [f"{non_auth_lines[0]} {non_auth_lines[1]}"] + non_auth_lines[2:]

    ordered_profiles = _catalog_header_order(blocks, start_idx, profiles) or profiles
    ordered_non_auth_profiles = [
        profile for profile in ordered_profiles
        if profile.get("role") != "identifier"
        and profile.get("semantic_type") not in {"catalog_title", "authorization_rule"}
    ]
    if non_auth_lines and ordered_non_auth_profiles:
        for idx, profile in enumerate(ordered_non_auth_profiles):
            if idx >= len(non_auth_lines):
                break
            value = non_auth_lines[idx] if idx < len(ordered_non_auth_profiles) - 1 else " ".join(non_auth_lines[idx:])
            record[str(profile.get("name"))] = value
    elif non_auth_lines:
        if len(non_auth_lines) >= 1:
            record["segmentacao_ans"] = non_auth_lines[0]
        if len(non_auth_lines) >= 2:
            record["cobertura_unimed_poa"] = non_auth_lines[1]
        if len(non_auth_lines) >= 3:
            record["emergencia"] = non_auth_lines[2]
        if len(non_auth_lines) >= 4:
            record["prazo_autorizacao_conforme_rn_n_623_ans"] = " ".join(non_auth_lines[3:])

    if auth_lines:
        record[auth_key] = " ".join(auth_lines)

    return record


def _artifact_catalog_field_value(record: dict[str, str], semantic_type: str, fallback_column: str) -> tuple[str, str] | None:
    candidates_by_semantic = {
        "deadline_rule": ["prazo_autorizacao_conforme_rn_n_623_ans", "prazo_autorizacao", "prazo"],
        "authorization_rule": ["orientacao_autorizacao_call_center", "orientacao_autorizacao", "autorizacao"],
        "coverage_rule": ["cobertura_unimed_poa", "cobertura"],
        "flag_boolean": ["emergencia", "urgencia"],
        "catalog_title": ["descricao_unimed_poa", "descricao", "titulo", "title"],
    }
    for key in candidates_by_semantic.get(semantic_type, []):
        value = str(record.get(key, "")).strip()
        if value:
            return key, value
    value = str(record.get(fallback_column, "")).strip()
    if value:
        return fallback_column, value
    return None


def _deterministic_catalog_field_answer(
    *,
    code: str,
    title: str,
    semantic_type: str,
    label: str,
    value: str,
) -> str:
    if semantic_type == "deadline_rule":
        return f"O prazo de autorizacao do procedimento {code} ({title}) e: {value}."
    if semantic_type == "authorization_rule":
        return f"A orientacao de autorizacao do procedimento {code} ({title}) e: {value}."
    if semantic_type == "coverage_rule":
        return f"A cobertura do procedimento {code} ({title}) e: {value}."
    if semantic_type == "flag_boolean":
        normalized_value = _normalize_for_match(value)
        if normalized_value in {"sim", "s"}:
            return f"O procedimento {code} ({title}) e classificado como emergencia."
        if normalized_value in {"nao", "n"}:
            return f"O procedimento {code} ({title}) nao e classificado como emergencia."
        return f"A informacao de emergencia do procedimento {code} ({title}) e: {value}."
    return f"O campo {label} do procedimento {code} ({title}) e: {value}."


def _catalog_prompt_answer_is_usable(
    *,
    answer: str,
    code: str,
    title: str,
    semantic_type: str,
    value: str,
) -> bool:
    text = str(answer or "").strip()
    if not text:
        return False
    normalized = _normalize_for_match(text)
    value_norm = _normalize_for_match(value)
    if normalized == value_norm:
        return False
    title_norm = _normalize_for_match(title)
    if title_norm and title_norm not in normalized:
        return False
    if code and code not in text and not title_norm:
        return False
    required_tokens = {
        "deadline_rule": ("prazo", "autoriz"),
        "authorization_rule": ("autoriz",),
        "coverage_rule": ("cobertura",),
    }.get(semantic_type, ())
    return all(token in normalized for token in required_tokens)


def _answer_catalog_code_from_artifact(
    *,
    collection: str,
    question: str,
    request_id: str,
    workspace_id: str,
) -> ChatResult | None:
    code = _extract_catalog_code(question)
    if not code:
        return None
    try:
        from src import controlplane
    except Exception:
        return None

    documents = controlplane.list_documents(workspace_id, collection)
    profiles = _catalog_profiles(workspace_id, collection)
    requested_field = _catalog_field_request(question, workspace_id, collection)
    for doc in documents:
        artifact = _resolve_processed_json(doc.doc_id, doc.filename)
        if not artifact:
            continue
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = payload.get("blocks", []) or []
        texts = [str(block.get("text", "")).strip() for block in blocks if str(block.get("text", "")).strip()]
        compact_blocks = [{"text": str(block.get("text", "")).strip()} for block in blocks if str(block.get("text", "")).strip()]
        for idx, text in enumerate(texts):
            if not re.search(rf"\b{re.escape(code)}\b", text):
                continue
            window = " ".join(part for part in texts[idx:idx + 3] if part).strip()
            compact = re.sub(r"\s+", " ", window)
            record = _extract_catalog_record_from_blocks(compact_blocks, idx, code, profiles=profiles)
            description = ""
            for candidate in [p.get("name") for p in profiles if p.get("semantic_type") == "catalog_title"] + ["descricao_unimed_poa", "descricao", "titulo", "title", "nome"]:
                if not candidate:
                    continue
                description = str(record.get(str(candidate), "")).strip()
                if description:
                    break
            if requested_field:
                target_column, semantic_type, label = requested_field
                field_value = _artifact_catalog_field_value(record, semantic_type, target_column)
                if field_value:
                    resolved_column, value = field_value
                    title = description or f"codigo {code}"
                    deterministic_answer = _deterministic_catalog_field_answer(
                        code=code,
                        title=title,
                        semantic_type=semantic_type,
                        label=label,
                        value=value,
                    )
                    answer = deterministic_answer
                    try:
                        record_json = json.dumps(record, ensure_ascii=False, indent=2)
                        prompt_messages = build_catalog_record_messages(
                            question=question,
                            target_column=resolved_column,
                            field_label=label,
                            record_json=record_json,
                        )
                        prompted_answer = llm.chat(
                            prompt_messages,
                            system=get_catalog_record_system(),
                        ).strip()
                        if _catalog_prompt_answer_is_usable(
                            answer=prompted_answer,
                            code=code,
                            title=title,
                            semantic_type=semantic_type,
                            value=value,
                        ):
                            answer = prompted_answer
                    except Exception:
                        answer = deterministic_answer
                    preview_parts: list[str] = []
                    if description:
                        preview_parts.append(description)
                    preview_parts.append(str(value).strip())
                    source_excerpt = re.sub(r"\s+", " ", " ".join(preview_parts)).strip() or compact[:400]
                    log_event(
                        logger,
                        20,
                        "Catalog field resolved from processed artifact using schema-aware fallback",
                        request_id=request_id,
                        collection=collection,
                        doc_id=doc.doc_id,
                        code=code,
                        target_column=resolved_column,
                        semantic_type=semantic_type,
                    )
                    return ChatResult(
                        answer=answer,
                        sources=[
                            Source(
                                chunk_id=f"artifact::{doc.doc_id}::{code}",
                                doc_id=doc.doc_id,
                                excerpt=source_excerpt[:400],
                                score=1.0,
                                metadata={
                                    "source_filename": doc.filename,
                                    "citation_label": "registro de catalogo (artefato prata)",
                                    "source_kind": "artifact_lookup",
                                    "artifact_record": record,
                                    "artifact_profiles": profiles,
                                    "target_column": resolved_column,
                                },
                            )
                        ],
                        request_id=request_id,
                    )
            answer = (
                f"Encontrei o codigo {code} no artefato processado."
                + (f" Trecho associado: {description}." if description else f" Contexto encontrado: {compact[:280]}.")
            )
            log_event(
                logger,
                20,
                "Catalog code resolved from processed artifact",
                request_id=request_id,
                collection=collection,
                doc_id=doc.doc_id,
                code=code,
            )
            return ChatResult(
                answer=answer,
                sources=[
                    Source(
                        chunk_id=f"artifact::{doc.doc_id}::{code}",
                        doc_id=doc.doc_id,
                        excerpt=compact[:400],
                        score=1.0,
                        metadata={
                            "source_filename": doc.filename,
                            "citation_label": "registro de catalogo (artefato prata)",
                            "source_kind": "artifact_lookup",
                        },
                    )
                ],
                request_id=request_id,
            )
    return None


def _artifact_header_level(text: str) -> tuple[str, str] | None:
    normalized = _normalize_for_match(text)
    for node_type, prefix in (("titulo", "titulo"), ("capitulo", "capitulo"), ("secao", "secao")):
        m = re.search(rf"\b{prefix}\s+([ivxlcdm]+|\d{{1,4}})\b", normalized)
        if m:
            return node_type, m.group(1)
    return None


def _structural_level(node_type: str) -> int:
    return {"titulo": 1, "capitulo": 2, "secao": 3, "subsecao": 4, "artigo": 5}.get(node_type, 99)


def _detect_requested_structural_type(question: str) -> str | None:
    q = _normalize_for_match(question)
    for token, node_type in (
        ("artigos", "artigo"),
        ("artigo", "artigo"),
        ("secoes", "secao"),
        ("secao", "secao"),
        ("capitulos", "capitulo"),
        ("capitulo", "capitulo"),
        ("titulos", "titulo"),
        ("titulo", "titulo"),
    ):
        if token in q:
            return node_type
    return None


def _extract_artifact_structures(blocks: list[dict]) -> list[dict]:
    headers: list[dict] = []
    for idx, block in enumerate(blocks):
        text = str(block.get("text", "")).strip()
        header = _artifact_header_level(text)
        if not header:
            continue
        headers.append(
            {
                "start_idx": idx,
                "node_type": header[0],
                "numeral": header[1].upper(),
                "label": text.splitlines()[0].strip(),
                "page_start": block.get("page_number"),
            }
        )

    structures: list[dict] = []
    for pos, header in enumerate(headers):
        end_idx = len(blocks)
        current_level = _structural_level(header["node_type"])
        for next_header in headers[pos + 1:]:
            if _structural_level(next_header["node_type"]) <= current_level:
                end_idx = next_header["start_idx"]
                break

        parts: list[str] = []
        articles: list[str] = []
        for block in blocks[header["start_idx"]:end_idx]:
            text = str(block.get("text", "")).strip()
            if not text:
                continue
            parts.append(text)
            for article in re.findall(r"Art\.?\s*(\d+)", text, re.IGNORECASE):
                label = f"Art. {article}"
                if label not in articles:
                    articles.append(label)

        structures.append(
            {
                **header,
                "end_idx": end_idx,
                "page_end": blocks[end_idx - 1].get("page_number") if end_idx > header["start_idx"] else header.get("page_start"),
                "text": "\n\n".join(parts).strip(),
                "articles": articles,
            }
        )
    return structures


def _extract_artifact_articles(blocks: list[dict]) -> list[dict]:
    articles: list[dict] = []
    current_article: dict | None = None
    current_context = {"titulo": "", "capitulo": "", "secao": ""}

    def _flush() -> None:
        nonlocal current_article
        if not current_article:
            return
        parts = current_article.pop("parts", [])
        current_article["text"] = "\n\n".join(part for part in parts if part).strip()
        articles.append(current_article)
        current_article = None

    for idx, block in enumerate(blocks):
        text = str(block.get("text", "")).strip()
        if not text:
            continue

        header = _artifact_header_level(text)
        if header:
            _flush()
            current_context[header[0]] = text.splitlines()[0].strip()
            lower_level = _structural_level(header[0])
            if lower_level <= _structural_level("capitulo"):
                current_context["secao"] = ""

        article_match = _ARTICLE_BLOCK_RE.search(text)
        if article_match:
            _flush()
            article_number = article_match.group(1)
            current_article = {
                "start_idx": idx,
                "end_idx": idx + 1,
                "article_number": article_number,
                "label": f"Art. {article_number}",
                "page_start": block.get("page_number"),
                "page_end": block.get("page_number"),
                "titulo": current_context.get("titulo", ""),
                "capitulo": current_context.get("capitulo", ""),
                "secao": current_context.get("secao", ""),
                "parts": [text],
            }
            continue

        if current_article:
            current_article["parts"].append(text)
            current_article["end_idx"] = idx + 1
            current_article["page_end"] = block.get("page_number") or current_article.get("page_end")

    _flush()
    return [article for article in articles if article.get("text")]


def _find_structure_scope(
    structures: list[dict],
    question: str,
) -> dict | None:
    targets = _extract_structural_targets(question)
    if not targets:
        return None
    for node_type, refs in targets:
        expanded_refs = _expand_refs_arabic_roman(refs)
        for item in structures:
            if item["node_type"] != node_type:
                continue
            if _expand_refs_arabic_roman({item["numeral"].lower()}) & expanded_refs:
                return item
    return None


def _find_exact_article_scope(blocks: list[dict], question: str) -> dict | None:
    article_refs = _extract_article_refs(question)
    if not article_refs:
        return None
    for article in _extract_artifact_articles(blocks):
        if article.get("article_number", "").lower() in article_refs:
            return article
    return None


def _find_contains_structural_pair(question: str) -> tuple[tuple[str, set[str]], tuple[str, set[str]]] | None:
    targets = _extract_structural_targets(question)
    if len(targets) < 2:
        return None

    typed = [(node_type, refs, _structural_level(node_type)) for node_type, refs in targets]
    typed = [item for item in typed if item[2] < 99]
    if len(typed) < 2:
        return None

    typed.sort(key=lambda item: item[2])
    parent_type, parent_refs, _ = typed[0]
    child_type, child_refs, _ = typed[-1]
    return (parent_type, parent_refs), (child_type, child_refs)


def _structure_is_within_scope(item: dict, scope: dict) -> bool:
    return (
        item["start_idx"] > scope["start_idx"]
        and item["start_idx"] < scope.get("end_idx", 10**9)
        and _structural_level(item["node_type"]) > _structural_level(scope["node_type"])
    )


def _answer_contains_structure(
    *,
    question: str,
    doc_label: str,
    structures: list[dict],
) -> tuple[str, dict] | None:
    pair = _find_contains_structural_pair(question)
    if not pair:
        return None
    (parent_type, parent_refs), (child_type, child_refs) = pair

    parent_scope = None
    for item in structures:
        if item["node_type"] != parent_type:
            continue
        if _expand_refs_arabic_roman({item["numeral"].lower()}) & _expand_refs_arabic_roman(parent_refs):
            parent_scope = item
            break
    if not parent_scope:
        return (
            f"Nao encontrei {parent_type} {next(iter(parent_refs), '')} no documento {doc_label}.",
            {"doc_label": doc_label, "requested_type": child_type},
        )

    if child_type == "artigo":
        normalized_articles = {
            re.search(r"(\d+)", label).group(1): label
            for label in parent_scope.get("articles", [])
            if re.search(r"(\d+)", label)
        }
        for ref in child_refs:
            if ref in normalized_articles:
                label = normalized_articles[ref]
                return (
                    f"Sim. {parent_scope['label']} contem {label}.",
                    {
                        "doc_label": doc_label,
                        "scope": parent_scope["label"],
                        "requested_type": child_type,
                        "child_label": label,
                    },
                )
        return (
            f"Nao. {parent_scope['label']} nao contem o artigo solicitado.",
            {"doc_label": doc_label, "scope": parent_scope["label"], "requested_type": child_type},
        )

    candidates = [
        item for item in structures
        if item["node_type"] == child_type and _structure_is_within_scope(item, parent_scope)
    ]
    expanded_child_refs = _expand_refs_arabic_roman(child_refs)
    for item in candidates:
        if _expand_refs_arabic_roman({item["numeral"].lower()}) & expanded_child_refs:
            return (
                f"Sim. {parent_scope['label']} contem {item['label']}.",
                {
                    "doc_label": doc_label,
                    "scope": parent_scope["label"],
                    "requested_type": child_type,
                    "child_label": item["label"],
                    "node_type": item["node_type"],
                    "page_number": item.get("page_start"),
                },
            )

    child_ref = next(iter(child_refs), "")
    return (
        f"Nao. {parent_scope['label']} nao contem {child_type} {child_ref.upper()}.",
        {"doc_label": doc_label, "scope": parent_scope["label"], "requested_type": child_type},
    )


def _build_structure_answer_text(
    *,
    intent: QueryIntent,
    requested_type: str,
    doc_label: str,
    structures: list[dict],
    scope: dict | None,
) -> tuple[str, dict]:
    target_items = [s for s in structures if s["node_type"] == requested_type]
    meta: dict = {"doc_label": doc_label}

    if intent == QueryIntent.COUNT_STRUCTURAL:
        if scope:
            if requested_type == "artigo":
                count = len(scope.get("articles", []))
            else:
                count = len(
                    [
                        s for s in structures
                        if s["node_type"] == requested_type
                        and s["start_idx"] > scope["start_idx"]
                        and s["start_idx"] < scope.get("end_idx", 10**9)
                        and _structural_level(s["node_type"]) > _structural_level(scope["node_type"])
                    ]
                )
            text = f"{scope['label']} possui {count} {requested_type}(s)."
            meta["scope"] = scope["label"]
            return text, meta

        count = len(target_items)
        text = f"O documento {doc_label} possui {count} {requested_type}(s)."
        if target_items:
            text += f" Vai de {target_items[0]['label']} ate {target_items[-1]['label']}."
        return text, meta

    if intent == QueryIntent.LIST_STRUCTURAL:
        if requested_type == "artigo" and scope:
            articles = scope.get("articles", [])
            meta["scope"] = scope["label"]
            if not articles:
                return f"Nao encontrei artigos explicitamente marcados dentro de {scope['label']}.", meta
            listed = ", ".join(articles[:30])
            return f"{scope['label']} cobre os seguintes artigos: {listed}.", meta

        if scope:
            child_level = _structural_level(requested_type)
            scope_level = _structural_level(scope["node_type"])
            scoped_items = [
                s for s in structures
                if s["node_type"] == requested_type
                and s["start_idx"] > scope["start_idx"]
                and s["start_idx"] < scope.get("end_idx", 10**9)
                and child_level > scope_level
            ]
            labels = [s["label"] for s in scoped_items[:20]]
            meta["scope"] = scope["label"]
            if not labels:
                return f"Nao encontrei {requested_type}(s) dentro de {scope['label']}.", meta
            return f"{scope['label']} contem {len(scoped_items)} {requested_type}(s): " + "; ".join(labels) + ".", meta

        labels = [s["label"] for s in target_items[:30]]
        if not labels:
            return f"Nao encontrei {requested_type}(s) no documento {doc_label}.", meta
        return f"O documento {doc_label} contem {len(target_items)} {requested_type}(s): " + "; ".join(labels) + ".", meta

    return "", meta


def _answer_structure_first(
    *,
    collection: str,
    question: str,
    request_id: str,
    intent: QueryIntent,
    workspace_id: str,
) -> ChatResult | None:
    from src import controlplane

    requested_type = _detect_requested_structural_type(question)
    if intent != QueryIntent.CONTAINS_STRUCTURAL and not requested_type:
        return None

    documents = controlplane.list_documents(workspace_id, collection)
    for doc in documents:
        artifact = _resolve_processed_json(doc.doc_id, doc.filename)
        if not artifact:
            continue
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = data.get("blocks", [])
        if not isinstance(blocks, list) or not blocks:
            continue
        structures = _extract_artifact_structures(blocks)
        if not structures:
            continue

        if intent == QueryIntent.CONTAINS_STRUCTURAL:
            contains_result = _answer_contains_structure(
                question=question,
                doc_label=doc.filename,
                structures=structures,
            )
            if not contains_result:
                continue
            text, meta = contains_result
            source = Source(
                chunk_id=f"artifact::{doc.doc_id}::contains",
                doc_id=doc.doc_id,
                excerpt=meta.get("scope", doc.filename),
                score=1.0,
                metadata=meta,
            )
            log_event(
                logger,
                20,
                "Structure-first contains query answered",
                request_id=request_id,
                collection=collection,
                intent=intent.value,
                doc_id=doc.doc_id,
                scope=meta.get("scope", ""),
                child_label=meta.get("child_label", ""),
            )
            metrics.increment("chat.structure_first.hit")
            return ChatResult(answer=text, sources=[source], request_id=request_id)

        scope = _find_structure_scope(structures, question)
        if intent == QueryIntent.LOCATE_STRUCTURAL:
            q_norm = _normalize_for_match(question)
            candidates = [s for s in structures if s["node_type"] == requested_type]
            if "ultimo" in q_norm or "ultima" in q_norm:
                scope = candidates[-1] if candidates else None
            elif "primeiro" in q_norm or "primeira" in q_norm:
                scope = candidates[0] if candidates else None
            if scope:
                answer = f"{scope['label']} aparece na pagina {scope.get('page_start') or '?'} do documento {doc.filename}."
                source = Source(
                    chunk_id=f"artifact::{doc.doc_id}::{scope['node_type']}::{scope['numeral']}",
                    doc_id=doc.doc_id,
                    excerpt=scope["label"],
                    score=1.0,
                    metadata={"node_type": scope["node_type"], "label": scope["label"], "page_number": scope.get("page_start")},
                )
                metrics.increment("chat.structure_first.hit")
                return ChatResult(answer=answer, sources=[source], request_id=request_id)

        text, meta = _build_structure_answer_text(
            intent=intent,
            requested_type=requested_type,
            doc_label=doc.filename,
            structures=structures,
            scope=scope,
        )
        if not text:
            continue
        source_meta = {"requested_type": requested_type, **meta}
        if scope:
            source_meta.update({"node_type": scope["node_type"], "label": scope["label"], "page_number": scope.get("page_start")})
        source = Source(
            chunk_id=f"artifact::{doc.doc_id}::{requested_type}",
            doc_id=doc.doc_id,
            excerpt=(scope["label"] if scope else doc.filename),
            score=1.0,
            metadata=source_meta,
        )
        log_event(
            logger,
            20,
            "Structure-first query answered",
            request_id=request_id,
            collection=collection,
            intent=intent.value,
            requested_type=requested_type,
            doc_id=doc.doc_id,
            scope=(scope["label"] if scope else ""),
        )
        metrics.increment("chat.structure_first.hit")
        return ChatResult(answer=text, sources=[source], request_id=request_id)

    metrics.increment("chat.structure_first.miss")
    return None


def _answer_exact_article_from_artifact(
    *,
    collection: str,
    question: str,
    history: list[ChatMessage] | None,
    request_id: str,
    workspace_id: str,
    profile_name: str,
) -> ChatResult | None:
    from src import controlplane

    article_refs = _extract_article_refs(question)
    if not article_refs:
        return None

    documents = controlplane.list_documents(workspace_id, collection)
    for doc in documents:
        artifact = _resolve_processed_json(doc.doc_id, doc.filename)
        if not artifact:
            continue
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = data.get("blocks", [])
        if not isinstance(blocks, list) or not blocks:
            continue

        article = _find_exact_article_scope(blocks, question)
        if not article:
            continue

        context = format_context(
            [
                {
                    "id": f"artifact::{doc.doc_id}::artigo::{article['article_number']}",
                    "text": article["text"],
                    "score": 1.0,
                    "metadata": {
                        "doc_id": doc.doc_id,
                        "artigo": article["label"],
                        "capitulo": article.get("capitulo", ""),
                        "secao": article.get("secao", ""),
                        "page_number": article.get("page_start"),
                        "chunk_type": "artifact_article",
                    },
                }
            ]
        )
        messages: list[dict] = [{"role": msg.role, "content": msg.content} for msg in (history or [])]
        messages.extend(build_rag_messages(context=context, question=question))
        answer_text = llm.chat(messages, system=get_rag_system(profile_name))

        source = Source(
            chunk_id=f"artifact::{doc.doc_id}::artigo::{article['article_number']}",
            doc_id=doc.doc_id,
            excerpt=article["text"][:400],
            score=1.0,
            metadata={
                "node_type": "artigo",
                "artigo": article["label"],
                "capitulo": article.get("capitulo", ""),
                "secao": article.get("secao", ""),
                "page_number": article.get("page_start"),
            },
        )
        log_event(
            logger,
            20,
            "Exact article resolved from processed artifact",
            request_id=request_id,
            collection=collection,
            doc_id=doc.doc_id,
            artigo=article["label"],
            capitulo=article.get("capitulo", ""),
        )
        metrics.increment("chat.article_exact.hit")
        return ChatResult(answer=answer_text, sources=[source], request_id=request_id)

    metrics.increment("chat.article_exact.miss")
    return None


def _extract_structural_scope_from_artifact(
    *,
    question: str,
    collection: str,
    workspace_id: str,
) -> dict | None:
    """Read processed JSON artifacts to resolve structural summaries deterministically."""
    from src import controlplane
    from src.summaries import generate_summary_from_scope_text

    targets = _extract_structural_targets(question)
    if not targets:
        refs = _extract_chapter_refs(question)
        if refs:
            targets = [("capitulo", refs)]
    if not targets:
        return None

    documents = controlplane.list_documents(workspace_id, collection)
    for doc in documents:
        artifact = _resolve_processed_json(doc.doc_id, doc.filename)
        if not artifact:
            continue
        try:
            data = json.loads(artifact.read_text(encoding="utf-8"))
        except Exception:
            continue
        blocks = data.get("blocks", [])
        if not isinstance(blocks, list) or not blocks:
            continue

        for node_type, refs in targets:
            expanded_refs = _expand_refs_arabic_roman(refs)
            for idx, block in enumerate(blocks):
                text = str(block.get("text", "")).strip()
                block_type = str(block.get("block_type", "")).strip().lower()
                if block_type not in {"section_header", "title", "header", "body"}:
                    continue
                header = _artifact_header_level(text)
                if not header or header[0] != node_type:
                    continue
                if not (_expand_refs_arabic_roman({header[1]}) & expanded_refs):
                    continue

                parts = [text]
                articles: list[str] = []
                current_level = {"titulo": 1, "capitulo": 2, "secao": 3}.get(node_type, 3)
                for next_block in blocks[idx + 1:]:
                    next_text = str(next_block.get("text", "")).strip()
                    next_header = _artifact_header_level(next_text)
                    if next_header:
                        next_level = {"titulo": 1, "capitulo": 2, "secao": 3}.get(next_header[0], 99)
                        if next_level <= current_level:
                            break
                    if next_text:
                        parts.append(next_text)
                        articles.extend(re.findall(r"Art\.?\s*(\d+)", next_text, re.IGNORECASE))

                merged = "\n\n".join(parts).strip()
                if not merged:
                    continue
                summary = generate_summary_from_scope_text(
                    node_id=f"artifact::{doc.doc_id}::{node_type}::{header[1]}",
                    node_type=node_type,
                    label=text.splitlines()[0].strip(),
                    path=text.splitlines()[0].strip(),
                    text=merged,
                    articles=articles,
                )
                out = summary.to_dict()
                out["doc_id"] = doc.doc_id
                log_event(
                    logger,
                    20,
                    "Structural summary resolved from processed artifact",
                    request_id="-",
                    collection=collection,
                    doc_id=doc.doc_id,
                    node_type=node_type,
                    label=out.get("label", ""),
                )
                metrics.increment("chat.summary.artifact_hit")
                return out
    return None


def _build_on_demand_summary_from_scope(
    *,
    question: str,
    scoped_results: list[dict],
    request_id: str,
    workspace_id: str,
    collection: str,
) -> dict | None:
    """Build summary on demand from already scoped chunks when cache is missing."""
    if not scoped_results:
        return None

    from src.summaries import generate_summary_from_scope_text

    ordered = sorted(
        scoped_results,
        key=lambda x: (
            x.get("metadata", {}).get("page_number", 10**9),
            x.get("metadata", {}).get("chunk_index", 10**9),
        ),
    )
    selected = ordered[:24]
    merged_text = "\n\n".join(str(item.get("text", "")).strip() for item in selected if str(item.get("text", "")).strip())
    if not merged_text.strip():
        return None

    anchor_meta = selected[0].get("metadata", {}) if selected else {}
    targets = _extract_structural_targets(question)
    node_type = targets[0][0] if targets else "capitulo"
    label = (
        str(anchor_meta.get(node_type, "")).strip()
        or str(anchor_meta.get("caminho_hierarquico", "")).strip()
        or f"Escopo {node_type}"
    )
    path = str(anchor_meta.get("caminho_hierarquico", "")).strip() or label
    node_id = str(anchor_meta.get("node_id", "")).strip() or f"on_demand::{hash(label + path) & 0xFFFFFFFF:08x}"
    article_values: list[str] = []
    for item in selected:
        art = str(item.get("metadata", {}).get("artigo", "")).strip()
        if art and art not in article_values:
            article_values.append(art)

    with metrics.time_block("chat.summary.on_demand_generation"):
        summary = generate_summary_from_scope_text(
            node_id=node_id,
            node_type=node_type,
            label=label,
            path=path,
            text=merged_text,
            articles=article_values,
        )
    metrics.increment("chat.summary.fallback_used")

    data = summary.to_dict()
    if data.get("status") == "invalid":
        log_event(
            logger,
            30,
            "On-demand structural summary invalid",
            request_id=request_id,
            node_type=node_type,
            label=label,
            errors=data.get("validation_errors", []),
        )
        return None

    log_event(
        logger,
        20,
        "On-demand structural summary generated",
        request_id=request_id,
        node_type=node_type,
        label=label,
        scoped_chunks=len(selected),
    )
    doc_id = str(anchor_meta.get("doc_id", "")).strip()
    if doc_id:
        from src import controlplane
        controlplane.upsert_document_summary(
            workspace_id=workspace_id,
            collection=collection,
            doc_id=doc_id,
            node_id=data.get("node_id", node_id),
            node_type=data.get("node_type", node_type),
            label=data.get("label", label),
            path=data.get("path", path),
            resumo_executivo=data.get("resumo_executivo", ""),
            resumo_juridico=data.get("resumo_juridico", ""),
            pontos_chave=data.get("pontos_chave", []),
            artigos_cobertos=data.get("artigos_cobertos", []),
            obrigacoes=data.get("obrigacoes", []),
            restricoes=data.get("restricoes", []),
            definicoes=data.get("definicoes", []),
            text_length=int(data.get("text_length", len(merged_text))),
            source_hash=str(data.get("source_hash", "")),
            source_text_length=int(data.get("source_text_length", len(merged_text))),
            status="fallback_only",
            invalid_reason="",
            generation_meta=data.get("generation_meta", {}),
        )
        metrics.increment("chat.summary.persisted_from_fallback")

    return data


# ── Main answer function ─────────────────────────────────────────────────────

def answer(
    collection: str,
    question: str,
    history: list[ChatMessage] | None = None,
    request_id: str = "-",
    where: dict | None = None,
    embedding_model: str | None = None,
    workspace_id: str = "default",
    domain_profile: str | None = None,
) -> ChatResult:
    """Semantic retrieval + LLM generation."""
    settings = get_settings()
    model_name = embedding_model or settings.embedding_model
    profile_name = domain_profile or settings.default_domain_profile
    physical_collection = vectordb.resolve_query_collection(collection, model_name, workspace_id=workspace_id)

    # ── Guardrails ────────────────────────────────────────────────
    question = sanitize_question(question)
    history = sanitize_history(history or [])

    injection = detect_injection(question)
    if injection:
        log_event(logger, 30, "Prompt injection detected in question", pattern=injection, request_id=request_id)
        return ChatResult(
            answer="Não foi possível processar essa pergunta. Reformule sua solicitação.",
            sources=[],
            request_id=request_id,
        )

    # ── Intent classification + summary shortcut ──────────────────
    intent = classify_query_intent(question)
    log_event(logger, 20, "Query intent classified", intent=intent.value, request_id=request_id)
    metrics.increment(f"chat.intent.{intent.value}")

    table_result = _answer_table_first(
        collection=collection,
        question=question,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    if table_result:
        return table_result

    artifact_catalog_result = _answer_catalog_code_from_artifact(
        collection=collection,
        question=question,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    if artifact_catalog_result:
        return artifact_catalog_result

    if intent in {
        QueryIntent.COUNT_STRUCTURAL,
        QueryIntent.LIST_STRUCTURAL,
        QueryIntent.LOCATE_STRUCTURAL,
        QueryIntent.CONTAINS_STRUCTURAL,
    }:
        structure_result = _answer_structure_first(
            collection=collection,
            question=question,
            request_id=request_id,
            intent=intent,
            workspace_id=workspace_id,
        )
        if structure_result:
            return structure_result

    if _extract_article_refs(question) and intent in {
        QueryIntent.QUESTION_STRUCTURAL,
        QueryIntent.LOCATE_EXCERPT,
        QueryIntent.QUESTION_FACTUAL,
    }:
        article_result = _answer_exact_article_from_artifact(
            collection=collection,
            question=question,
            history=history,
            request_id=request_id,
            workspace_id=workspace_id,
            profile_name=profile_name,
        )
        if article_result:
            return article_result

    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        summary = _resolve_structural_summary(question, collection, workspace_id)
        if _summary_is_usable(summary):
            log_event(logger, 20, "Summary structural shortcut activated",
                      label=summary.get("label", ""), request_id=request_id)
            return _answer_from_summary(
                summary=summary,
                question=question,
                history=history,
                request_id=request_id,
                profile_name=profile_name,
                collection=collection,
                physical_collection=physical_collection,
                workspace_id=workspace_id,
                model_name=model_name,
            )
        artifact_summary = _extract_structural_scope_from_artifact(
            question=question,
            collection=collection,
            workspace_id=workspace_id,
        )
        if _summary_is_usable(artifact_summary):
            log_event(
                logger,
                20,
                "Summary structural resolved from processed artifact",
                request_id=request_id,
                label=artifact_summary.get("label", ""),
            )
            return _answer_from_summary(
                summary=artifact_summary,
                question=question,
                history=history,
                request_id=request_id,
                profile_name=profile_name,
                collection=collection,
                physical_collection=physical_collection,
                workspace_id=workspace_id,
                model_name=model_name,
            )
        if summary and not _summary_is_usable(summary):
            log_event(
                logger,
                30,
                "Pre-computed summary ignored due to invalid/empty payload",
                request_id=request_id,
                label=summary.get("label", ""),
                status=summary.get("status", ""),
            )
        log_event(
            logger,
            20,
            "No pre-computed summary found; using strict chapter scope in retrieval",
            request_id=request_id,
        )

    # ── Retrieval ─────────────────────────────────────────────────
    fused = _retrieve_with_routing(
        question=question,
        collection=collection,
        physical_collection=physical_collection,
        top_k=settings.retrieval_top_k,
        settings=settings,
        where=where,
        model_name=model_name,
        request_id=request_id,
    )
    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        scoped = _metadata_structural_scope_from_seed(
            physical_collection=physical_collection,
            question=question,
            seed_results=fused,
        )
        if scoped:
            fused = scoped
            log_event(
                logger,
                20,
                "Metadata structural scope resolved for summary",
                request_id=request_id,
                retained=len(fused),
            )
    fused = _supplement_chapter_matches(fused, physical_collection, question=question)
    fused = _expand_exact_chapter_context(fused, physical_collection, question=question)

    if not fused:
        log_event(
            logger, 20, "No retrieval results found",
            collection=collection,
            physical_collection=physical_collection,
            request_id=request_id,
        )
        return ChatResult(
            answer="Não encontrei informações relevantes nos documentos disponíveis para essa coleção.",
            sources=[],
            request_id=request_id,
        )

    # ── Parent expansion for legal chunks ─────────────────────────
    max_exp = int(settings.max_context_tokens * 0.4)
    fused = _expand_legal_context(
        fused, physical_collection,
        max_expansion_tokens=max_exp,
        chars_per_token=settings.chars_per_token,
    )
    fused = _expand_adjacent_structural_context(
        fused,
        physical_collection,
        question=question,
        window=getattr(settings, "retrieval_adjacency_window", 1),
    )
    fused = _prioritize_exact_chapter_matches(fused, question=question)
    fused = _boost_section_hint_compatibility(
        fused,
        question=question,
        bonus=getattr(settings, "retrieval_section_hint_bonus", 0.14),
    )
    fused = _rerank_structural_continuity(
        fused,
        question=question,
        bonus=getattr(settings, "retrieval_structural_bonus", 0.0) * 0.5,
    )
    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        scoped, scoped_ok = _enforce_summary_structural_scope(fused, question=question, min_hits=1)
        if scoped_ok:
            fused = scoped
            log_event(
                logger,
                20,
                "Summary structural scope enforced",
                request_id=request_id,
                retained=len(fused),
            )
        else:
            return ChatResult(
                answer="Nao localizei trechos suficientes da estrutura solicitada (titulo/capitulo/secao/artigo) nos documentos desta colecao.",
                sources=[],
                request_id=request_id,
            )
        ondemand_summary = _build_on_demand_summary_from_scope(
            question=question,
            scoped_results=fused,
            request_id=request_id,
            workspace_id=workspace_id,
            collection=collection,
        )
        if ondemand_summary:
            result = _answer_from_summary(
                summary=ondemand_summary,
                question=question,
                history=history,
                request_id=request_id,
                profile_name=profile_name,
                collection=collection,
                physical_collection=physical_collection,
                workspace_id=workspace_id,
                model_name=model_name,
            )
            return result
        ondemand_summary = _build_on_demand_summary_from_scope(
            question=question,
            scoped_results=fused,
            request_id=request_id,
            workspace_id=workspace_id,
            collection=collection,
        )
        if ondemand_summary:
            return _answer_from_summary(
                summary=ondemand_summary,
                question=question,
                history=history,
                request_id=request_id,
                profile_name=profile_name,
                collection=collection,
                physical_collection=physical_collection,
                workspace_id=workspace_id,
                model_name=model_name,
            )

    # ── Sanitize + compress + trim + format ────────────────────────
    for item in fused:
        item["text"] = sanitize_context_chunk(item["text"])

    if getattr(settings, "compression_enabled", False):
        from src.compressor import compress_chunks

        fused = compress_chunks(
            query=question,
            chunks=fused,
            method=getattr(settings, "compression_method", "extractive"),
            max_sentences=getattr(settings, "compression_max_sentences", 3),
        )

    fused = _trim_to_budget(fused, settings.max_context_tokens, settings.chars_per_token)
    context = format_context(fused)
    context_tokens = _estimate_tokens(context, settings.chars_per_token)
    metrics.observe("chat.context_tokens", context_tokens)

    # ── LLM generation ────────────────────────────────────────────
    messages: list[dict] = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.extend(build_rag_messages(context=context, question=question))

    with metrics.time_block("chat.llm_generation"):
        answer_text = llm.chat(messages, system=get_rag_system(profile_name))
    answer_text = _mark_possible_contradiction(answer_text, fused, question)

    sources = [
        Source(
            chunk_id=item["id"],
            doc_id=item.get("metadata", {}).get("doc_id", ""),
            excerpt=_build_source_excerpt(item),
            score=item["score"],
            metadata=item.get("metadata", {}),
        )
        for item in fused[:5]
    ]

    log_event(
        logger, 20, "Chat answered",
        collection=collection,
        sources=len(sources),
        request_id=request_id,
        domain_profile=profile_name,
        context_tokens=context_tokens,
    )
    return ChatResult(answer=answer_text, sources=sources, request_id=request_id)


# ── Streaming ────────────────────────────────────────────────────────────────

@dataclass
class StreamContext:
    """Holds retrieval results for streaming — sources are sent before the LLM tokens."""
    sources: list[Source]
    request_id: str
    messages: list[dict]
    system: str


def prepare_stream(
    collection: str,
    question: str,
    history: list[ChatMessage] | None = None,
    request_id: str = "-",
    where: dict | None = None,
    embedding_model: str | None = None,
    workspace_id: str = "default",
    domain_profile: str | None = None,
) -> StreamContext | ChatResult:
    """Run retrieval and return context for streaming, or ChatResult if blocked/empty."""
    settings = get_settings()
    model_name = embedding_model or settings.embedding_model
    profile_name = domain_profile or settings.default_domain_profile
    physical_collection = vectordb.resolve_query_collection(collection, model_name, workspace_id=workspace_id)

    question = sanitize_question(question)
    history = sanitize_history(history or [])

    injection = detect_injection(question)
    if injection:
        log_event(logger, 30, "Prompt injection detected in question", pattern=injection, request_id=request_id)
        return ChatResult(
            answer="Não foi possível processar essa pergunta. Reformule sua solicitação.",
            sources=[],
            request_id=request_id,
        )

    # ── Intent classification + summary shortcut for streaming ────
    intent = classify_query_intent(question)
    log_event(logger, 20, "Query intent classified",
              intent=intent.value, request_id=request_id)

    table_result = _answer_table_first(
        collection=collection,
        question=question,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    if table_result:
        return table_result

    artifact_catalog_result = _answer_catalog_code_from_artifact(
        collection=collection,
        question=question,
        request_id=request_id,
        workspace_id=workspace_id,
    )
    if artifact_catalog_result:
        return artifact_catalog_result

    if intent in {
        QueryIntent.COUNT_STRUCTURAL,
        QueryIntent.LIST_STRUCTURAL,
        QueryIntent.LOCATE_STRUCTURAL,
        QueryIntent.CONTAINS_STRUCTURAL,
    }:
        structure_result = _answer_structure_first(
            collection=collection,
            question=question,
            request_id=request_id,
            intent=intent,
            workspace_id=workspace_id,
        )
        if structure_result:
            return structure_result

    if _extract_article_refs(question) and intent in {
        QueryIntent.QUESTION_STRUCTURAL,
        QueryIntent.LOCATE_EXCERPT,
        QueryIntent.QUESTION_FACTUAL,
    }:
        article_result = _answer_exact_article_from_artifact(
            collection=collection,
            question=question,
            history=history,
            request_id=request_id,
            workspace_id=workspace_id,
            profile_name=profile_name,
        )
        if article_result:
            return article_result

    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        summary = _resolve_structural_summary(question, collection, workspace_id)
        if _summary_is_usable(summary):
            log_event(logger, 20, "Summary structural shortcut activated",
                      label=summary.get("label", ""), request_id=request_id)
            result = _answer_from_summary(
                summary=summary,
                question=question,
                history=history,
                request_id=request_id,
                profile_name=profile_name,
                collection=collection,
                physical_collection=physical_collection,
                workspace_id=workspace_id,
                model_name=model_name,
            )
            return result
        artifact_summary = _extract_structural_scope_from_artifact(
            question=question,
            collection=collection,
            workspace_id=workspace_id,
        )
        if _summary_is_usable(artifact_summary):
            log_event(
                logger,
                20,
                "Streaming summary structural resolved from processed artifact",
                request_id=request_id,
                label=artifact_summary.get("label", ""),
            )
            result = _answer_from_summary(
                summary=artifact_summary,
                question=question,
                history=history,
                request_id=request_id,
                profile_name=profile_name,
                collection=collection,
                physical_collection=physical_collection,
                workspace_id=workspace_id,
                model_name=model_name,
            )
            return result
        if summary and not _summary_is_usable(summary):
            log_event(
                logger,
                30,
                "Pre-computed summary ignored due to invalid/empty payload",
                request_id=request_id,
                label=summary.get("label", ""),
                status=summary.get("status", ""),
            )
        log_event(
            logger,
            20,
            "No pre-computed summary found; using strict chapter scope in streaming retrieval",
            request_id=request_id,
        )

    fused = _retrieve_with_routing(
        question=question,
        collection=collection,
        physical_collection=physical_collection,
        top_k=settings.retrieval_top_k,
        settings=settings,
        where=where,
        model_name=model_name,
        request_id=request_id,
    )
    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        scoped = _metadata_structural_scope_from_seed(
            physical_collection=physical_collection,
            question=question,
            seed_results=fused,
        )
        if scoped:
            fused = scoped
            log_event(
                logger,
                20,
                "Metadata structural scope resolved for streaming summary",
                request_id=request_id,
                retained=len(fused),
            )
    fused = _supplement_chapter_matches(fused, physical_collection, question=question)
    fused = _expand_exact_chapter_context(fused, physical_collection, question=question)

    if not fused:
        return ChatResult(
            answer="Não encontrei informações relevantes nos documentos disponíveis para essa coleção.",
            sources=[],
            request_id=request_id,
        )

    # Parent expansion for legal chunks
    max_exp = int(settings.max_context_tokens * 0.4)
    fused = _expand_legal_context(
        fused, physical_collection,
        max_expansion_tokens=max_exp,
        chars_per_token=settings.chars_per_token,
    )
    fused = _expand_adjacent_structural_context(
        fused,
        physical_collection,
        question=question,
        window=getattr(settings, "retrieval_adjacency_window", 1),
    )
    fused = _prioritize_exact_chapter_matches(fused, question=question)
    fused = _boost_section_hint_compatibility(
        fused,
        question=question,
        bonus=getattr(settings, "retrieval_section_hint_bonus", 0.14),
    )
    fused = _rerank_structural_continuity(
        fused,
        question=question,
        bonus=getattr(settings, "retrieval_structural_bonus", 0.0) * 0.5,
    )
    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        scoped, scoped_ok = _enforce_summary_structural_scope(fused, question=question, min_hits=1)
        if scoped_ok:
            fused = scoped
            log_event(
                logger,
                20,
                "Summary structural scope enforced in streaming flow",
                request_id=request_id,
                retained=len(fused),
            )
        else:
            return ChatResult(
                answer="Nao localizei trechos suficientes da estrutura solicitada (titulo/capitulo/secao/artigo) nos documentos desta colecao.",
                sources=[],
                request_id=request_id,
            )

    for item in fused:
        item["text"] = sanitize_context_chunk(item["text"])

    if getattr(settings, "compression_enabled", False):
        from src.compressor import compress_chunks

        fused = compress_chunks(
            query=question,
            chunks=fused,
            method=getattr(settings, "compression_method", "extractive"),
            max_sentences=getattr(settings, "compression_max_sentences", 3),
        )

    fused = _trim_to_budget(fused, settings.max_context_tokens, settings.chars_per_token)
    context = format_context(fused)

    messages: list[dict] = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.extend(build_rag_messages(context=context, question=question))

    sources = [
        Source(
            chunk_id=item["id"],
            doc_id=item.get("metadata", {}).get("doc_id", ""),
            excerpt=_build_source_excerpt(item),
            score=item["score"],
            metadata=item.get("metadata", {}),
        )
        for item in fused[:5]
    ]

    return StreamContext(
        sources=sources,
        request_id=request_id,
        messages=messages,
        system=get_rag_system(profile_name),
    )

