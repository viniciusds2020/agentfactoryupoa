"""Semantic retrieval (vector search) and RAG generation."""
from __future__ import annotations

import re
import unicodedata
import json
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from src import llm, vectordb
from src.config import get_settings
from src.guardrails import detect_injection, sanitize_context_chunk, sanitize_history, sanitize_question
from src.lexical import normalize_query_numerals
from src.observability import metrics
from src.prompts import build_rag_messages, format_context, get_rag_system
from src.table_renderer import render_table_answer
from src.table_semantics import aggregation_lead_text, infer_subject_label, render_value_by_unit
from src.utils import get_logger, log_event

logger = get_logger(__name__)


# â”€â”€ Context budget management â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Data classes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


def _normalize_for_match(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.lower()


def _keyword_tokens(text: str) -> list[str]:
    normalized = _normalize_for_match(normalize_query_numerals(text))
    raw_tokens = re.findall(r"\b[a-z0-9]+\b", normalized)
    stopwords = {
        "dos", "das", "para", "com", "sem", "que", "uma", "por", "art", "artigo",
        "secao", "capitulo", "titulo", "sobre", "quais", "qual", "sao", "sao", "sua",
    }
    tokens: list[str] = []
    seen: set[str] = set()
    for tok in raw_tokens:
        if tok in stopwords:
            continue
        # Keep semantic words (>=3 chars) and structural numerals (1, I, IV, 12...).
        if len(tok) >= 3 or tok.isdigit() or re.fullmatch(r"[ivxlcdm]{1,8}", tok):
            if tok not in seen:
                tokens.append(tok)
                seen.add(tok)
    return tokens


def _metadata_structural_text(meta: dict) -> str:
    return " | ".join(
        str(meta.get(key, "")).strip()
        for key in ("titulo", "capitulo", "secao", "subsection", "section", "artigo", "paragrafo", "inciso", "caminho_hierarquico")
        if str(meta.get(key, "")).strip()
    )


def _is_structural_query(question: str) -> bool:
    q = _normalize_for_match(question)
    return any(term in q for term in ("capitulo", "secao", "artigo", "titulo", "inciso", "paragrafo"))


_CHAPTER_REF_RE = re.compile(r"\bcapitulo\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_TITLE_REF_RE = re.compile(r"\btitulo\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"\bsecao\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_SUBSECTION_REF_RE = re.compile(r"\bsubsecao\s+([ivxlcdm]+|\d{1,4})\b", re.IGNORECASE)
_ARTICLE_REF_RE = re.compile(r"\bart(?:igo)?\.?\s*([0-9]{1,4})(?:[º°o])?\b", re.IGNORECASE)
_ARTICLE_BLOCK_RE = re.compile(r"\bArt\.?\s*(\d{1,4})(?:[º°o])?\b", re.IGNORECASE)

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


def _extract_chapter_refs(question: str) -> set[str]:
    expanded = _normalize_for_match(normalize_query_numerals(question))
    refs: set[str] = set()
    for match in _CHAPTER_REF_RE.finditer(expanded):
        refs.add(match.group(1).lower())
    return refs


def _extract_article_refs(question: str) -> set[str]:
    expanded = _normalize_for_match(normalize_query_numerals(question))
    return {match.group(1).lower() for match in _ARTICLE_REF_RE.finditer(expanded)}


def _extract_structural_targets(question: str) -> list[tuple[str, set[str]]]:
    """Extract structural targets from query (titulo/capitulo/secao/subsecao/artigo)."""
    expanded = _normalize_for_match(normalize_query_numerals(question))
    targets: list[tuple[str, set[str]]] = []
    for node_type, pattern in (
        ("titulo", _TITLE_REF_RE),
        ("capitulo", _CHAPTER_REF_RE),
        ("secao", _SECTION_REF_RE),
        ("subsecao", _SUBSECTION_REF_RE),
        ("artigo", _ARTICLE_REF_RE),
    ):
        refs: set[str] = set()
        for match in pattern.finditer(expanded):
            refs.add(match.group(1).lower())
        if refs:
            targets.append((node_type, refs))
    return targets


def _extract_chapter_refs_from_text(text: str) -> set[str]:
    normalized = _normalize_for_match(normalize_query_numerals(text))
    refs: set[str] = set()
    for match in _CHAPTER_REF_RE.finditer(normalized):
        refs.add(match.group(1).lower())
    return refs


def _expand_refs_arabic_roman(refs: set[str]) -> set[str]:
    """Expand chapter refs to include both Arabic and Roman numeral forms.

    E.g., {"2"} â†’ {"2", "ii"}, {"ii"} â†’ {"ii", "2"}, {"iv"} â†’ {"iv", "4"}
    """
    from src.lexical import int_to_roman, roman_to_int
    expanded = set(refs)
    for ref in refs:
        if ref.isdigit():
            n = int(ref)
            if 1 <= n <= 3999:
                expanded.add(int_to_roman(n).lower())
        elif ref.isalpha():
            n = roman_to_int(ref.upper())
            if n > 0:
                expanded.add(str(n))
    return expanded


def _item_matches_chapter_refs(item: dict, refs: set[str]) -> bool:
    if not refs:
        return False
    meta = item.get("metadata", {})
    structural = _metadata_structural_text(meta)
    if not structural:
        return False
    item_refs = _expand_refs_arabic_roman(_extract_chapter_refs_from_text(structural))
    expanded_refs = _expand_refs_arabic_roman(refs)
    return bool(item_refs & expanded_refs)


def _extract_refs_from_meta_value(value: str) -> set[str]:
    return _expand_refs_arabic_roman(_extract_chapter_refs_from_text(value))


def _extract_node_refs_from_text(value: str, node_type: str) -> set[str]:
    normalized = _normalize_for_match(normalize_query_numerals(value))
    if node_type == "artigo":
        return set(re.findall(r"\b\d{1,4}\b", normalized))
    pattern = _STRUCTURAL_LABEL_REF_RE.get(node_type)
    refs: set[str] = set()
    if pattern:
        for match in pattern.finditer(normalized):
            refs.add(match.group(1).lower())
    return _expand_refs_arabic_roman(refs)


def _item_matches_structural_target(item: dict, node_type: str, refs: set[str]) -> bool:
    if not refs:
        return False
    meta = item.get("metadata", {}) or {}
    keys = _NODE_META_KEYS.get(node_type, ())
    expanded_refs = _expand_refs_arabic_roman(refs)
    for key in keys:
        value = str(meta.get(key, "")).strip()
        if not value:
            continue
        if node_type == "artigo":
            value_tokens = set(re.findall(r"\d{1,4}", _normalize_for_match(value)))
            if value_tokens & expanded_refs:
                return True
            continue
        meta_refs = _extract_node_refs_from_text(value, node_type)
        if meta_refs & expanded_refs:
            return True
        # Robust fallback: some OCR/parsing variants may corrupt labels
        # (e.g., "CAPÃTULO"), but numerals still survive.
        generic_num_tokens = set(
            re.findall(r"\b(?:[ivxlcdm]+|\d{1,4})\b", _normalize_for_match(value))
        )
        if _expand_refs_arabic_roman(generic_num_tokens) & expanded_refs:
            return True
    structural = _metadata_structural_text(meta)
    if structural:
        structural_refs = _extract_refs_from_meta_value(structural)
        if structural_refs & expanded_refs:
            return True
    return False


def _expand_exact_chapter_context(
    results: list[dict],
    physical_collection: str,
    question: str,
    max_docs: int = 3,
    max_add: int = 24,
) -> list[dict]:
    refs = _extract_chapter_refs(question)
    if not results or not refs:
        return results

    anchors = [item for item in results if _item_matches_chapter_refs(item, refs)]
    if not anchors:
        return results

    seen_ids = {item["id"] for item in results}
    expanded = list(results)
    doc_ids: list[str] = []
    for item in anchors:
        doc_id = str(item.get("metadata", {}).get("doc_id", "")).strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)

    best_anchor_score = max((float(item.get("score", 0.0)) for item in anchors), default=0.0)
    added = 0
    for doc_id in doc_ids[:max_docs]:
        doc_items = vectordb.get_by_metadata(physical_collection, {"doc_id": {"$eq": doc_id}})
        doc_items = sorted(doc_items, key=lambda x: x.get("metadata", {}).get("chunk_index", 0))
        for item in doc_items:
            if item.get("id") in seen_ids:
                continue
            if not _item_matches_chapter_refs(item, refs):
                continue
            enriched = dict(item)
            enriched["score"] = max(best_anchor_score * 0.97, float(item.get("score", 0.0)))
            expanded.append(enriched)
            seen_ids.add(enriched["id"])
            added += 1
            if added >= max_add:
                return expanded
    return expanded


def _prioritize_exact_chapter_matches(results: list[dict], question: str) -> list[dict]:
    refs = _extract_chapter_refs(question)
    if not results or not refs:
        return results

    with_match: list[dict] = []
    without_match: list[dict] = []
    for item in results:
        adjusted = dict(item)
        if _item_matches_chapter_refs(adjusted, refs):
            bonus = 0.32
            if adjusted.get("metadata", {}).get("chunk_type") == "parent":
                bonus += 0.08
            adjusted["score"] = float(adjusted.get("score", 0.0)) + bonus
            with_match.append(adjusted)
        else:
            without_match.append(adjusted)

    if not with_match:
        return results

    with_match = sorted(with_match, key=lambda x: x.get("score", 0.0), reverse=True)
    without_match = sorted(without_match, key=lambda x: x.get("score", 0.0), reverse=True)
    return with_match + without_match


def _metadata_hint_text(meta: dict) -> str:
    return " | ".join(
        str(meta.get(key, "")).strip()
        for key in (
            "pdf_section_hints",
            "section_hint",
            "titulo",
            "capitulo",
            "secao",
            "subsection",
            "section",
            "caminho_hierarquico",
            "artigo",
        )
        if str(meta.get(key, "")).strip()
    )


def _boost_section_hint_compatibility(results: list[dict], question: str, bonus: float = 0.14) -> list[dict]:
    """Boost chunks whose section hints/structural metadata match the query intent."""
    if not results or bonus <= 0 or not _is_structural_query(question):
        return results

    refs = _extract_chapter_refs(question)
    tokens = _keyword_tokens(question)
    boosted: list[dict] = []

    for item in results:
        adjusted = dict(item)
        score = float(adjusted.get("score", 0.0))
        meta = adjusted.get("metadata", {}) or {}
        hint_text = _normalize_for_match(_metadata_hint_text(meta))

        if refs and _item_matches_chapter_refs(adjusted, refs):
            score += bonus * 2.0

        if hint_text and tokens:
            overlap = 0
            for tok in tokens:
                if tok.isdigit() or re.fullmatch(r"[ivxlcdm]{1,8}", tok):
                    if re.search(rf"\b{re.escape(tok)}\b", hint_text):
                        overlap += 1
                elif tok in hint_text:
                    overlap += 1
            if overlap:
                score += min(overlap, 3) * bonus * 0.45

        adjusted["score"] = score
        boosted.append(adjusted)

    return sorted(boosted, key=lambda x: x.get("score", 0.0), reverse=True)


def _enforce_summary_structural_scope(
    results: list[dict],
    question: str,
    min_hits: int = 1,
) -> tuple[list[dict], bool]:
    """Keep only chunks that match the requested structural target."""
    targets = _extract_structural_targets(question)
    if not targets:
        refs = _extract_chapter_refs(question)
        targets = [("capitulo", refs)] if refs else []
    if not targets:
        return results, False
    matched = [
        item
        for item in results
        if any(_item_matches_structural_target(item, node_type, refs) for node_type, refs in targets)
    ]
    if len(matched) < min_hits:
        return results, False
    matched = sorted(matched, key=lambda x: x.get("score", 0.0), reverse=True)
    return matched, True


def _metadata_structural_scope_from_seed(
    *,
    physical_collection: str,
    question: str,
    seed_results: list[dict],
    max_docs: int = 3,
    max_items: int = 80,
) -> list[dict]:
    """Resolve structural scope directly from metadata for top seeded documents."""
    targets = _extract_structural_targets(question)
    if not targets:
        refs = _extract_chapter_refs(question)
        if refs:
            targets = [("capitulo", refs)]
    if not targets or not seed_results:
        return []

    doc_ids: list[str] = []
    for item in seed_results:
        doc_id = str(item.get("metadata", {}).get("doc_id", "")).strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    if not doc_ids:
        return []

    seeded_by_doc: dict[str, float] = {}
    for item in seed_results:
        doc_id = str(item.get("metadata", {}).get("doc_id", "")).strip()
        if not doc_id:
            continue
        seeded_by_doc[doc_id] = max(seeded_by_doc.get(doc_id, 0.0), float(item.get("score", 0.0)))

    scoped: list[dict] = []
    seen_ids: set[str] = set()
    for doc_id in doc_ids[:max_docs]:
        doc_items = vectordb.get_by_metadata(physical_collection, {"doc_id": {"$eq": doc_id}})
        for item in doc_items:
            if item.get("id") in seen_ids:
                continue
            if not any(_item_matches_structural_target(item, node_type, refs) for node_type, refs in targets):
                continue
            adjusted = dict(item)
            base = seeded_by_doc.get(doc_id, 0.0)
            # Metadata-resolved scope should dominate summary_structural flow.
            adjusted["score"] = max(float(adjusted.get("score", 0.0)), base + 0.4)
            scoped.append(adjusted)
            seen_ids.add(adjusted["id"])
            if len(scoped) >= max_items:
                break
        if len(scoped) >= max_items:
            break

    scoped = sorted(
        scoped,
        key=lambda x: (
            x.get("metadata", {}).get("page_number", 10**9),
            x.get("metadata", {}).get("chunk_index", 10**9),
        ),
    )
    return scoped


def _supplement_chapter_matches(
    results: list[dict],
    physical_collection: str,
    question: str,
    max_docs: int = 3,
    max_add: int = 8,
) -> list[dict]:
    refs = _extract_chapter_refs(question)
    if not results or not refs:
        return results

    # Expand refs to include both Arabic and Roman forms
    expanded_refs = _expand_refs_arabic_roman(refs)

    doc_ids: list[str] = []
    for item in results:
        doc_id = str(item.get("metadata", {}).get("doc_id", "")).strip()
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    if not doc_ids:
        return results

    boosted = list(results)
    seen_ids = {item["id"] for item in boosted}
    best_score = max((float(item.get("score", 0.0)) for item in boosted), default=0.0)
    added = 0

    for doc_id in doc_ids[:max_docs]:
        doc_items = vectordb.get_by_metadata(physical_collection, {"doc_id": {"$eq": doc_id}})
        for item in doc_items:
            if item.get("id") in seen_ids:
                continue
            structural = _normalize_for_match(_metadata_structural_text(item.get("metadata", {})))
            if not structural:
                continue
            if any(re.search(rf"\bcapitulo\s+{re.escape(ref)}\b", structural) for ref in expanded_refs):
                enriched = dict(item)
                enriched["score"] = max(best_score + 0.08, float(item.get("score", 0.0)))
                boosted.append(enriched)
                seen_ids.add(enriched["id"])
                added += 1
                if added >= max_add:
                    return boosted

    return boosted


def _structural_match_strength(question: str, item: dict) -> int:
    meta = item.get("metadata", {})
    structural_text = _normalize_for_match(_metadata_structural_text(meta))
    if not structural_text:
        return 0
    tokens = _keyword_tokens(question)
    if not tokens:
        return 0
    strength = 0
    for tok in tokens:
        if tok.isdigit() or re.fullmatch(r"[ivxlcdm]{1,8}", tok):
            # Exact boundary match avoids false positives like "1" matching "11".
            if re.search(rf"\b{re.escape(tok)}\b", structural_text):
                strength += 1
        elif tok in structural_text:
            strength += 1
    return strength


def _rerank_structural_continuity(results: list[dict], question: str, bonus: float) -> list[dict]:
    if not results or bonus <= 0:
        return results

    parent_key_counts: dict[str, int] = {}
    for item in results:
        parent_key = item.get("metadata", {}).get("parent_key", "")
        if parent_key:
            parent_key_counts[parent_key] = parent_key_counts.get(parent_key, 0) + 1

    reranked: list[dict] = []
    for item in results:
        meta = item.get("metadata", {})
        adjusted = dict(item)
        adjusted["base_score"] = item.get("score", 0.0)
        adjusted_score = item.get("score", 0.0)

        strength = _structural_match_strength(question, item)
        if strength:
            adjusted_score += min(strength, 3) * bonus

        parent_key = meta.get("parent_key", "")
        if parent_key and parent_key_counts.get(parent_key, 0) > 1:
            adjusted_score += bonus * 0.6

        if meta.get("chunk_type") == "parent":
            adjusted_score += bonus * 0.35

        adjusted["score"] = adjusted_score
        reranked.append(adjusted)

    return sorted(reranked, key=lambda x: x["score"], reverse=True)


def _expand_adjacent_structural_context(
    results: list[dict],
    physical_collection: str,
    question: str,
    window: int = 1,
    max_docs: int = 3,
) -> list[dict]:
    """Pull neighboring chunks when the retrieved chunk strongly matches a section title."""
    if not results or window <= 0:
        return results

    anchors = [item for item in results if _structural_match_strength(question, item) > 0]
    if not anchors:
        return results

    seen_ids = {item["id"] for item in results}
    expanded = list(results)
    doc_cache: dict[str, list[dict]] = {}

    for item in anchors[:max_docs]:
        meta = item.get("metadata", {})
        doc_id = meta.get("doc_id", "")
        chunk_index = meta.get("chunk_index")
        if not doc_id or chunk_index is None:
            continue

        if doc_id not in doc_cache:
            doc_items = vectordb.get_by_metadata(physical_collection, {"doc_id": {"$eq": doc_id}})
            doc_cache[doc_id] = sorted(
                doc_items,
                key=lambda x: x.get("metadata", {}).get("chunk_index", 0),
            )

        neighbors_by_index = {
            doc_item.get("metadata", {}).get("chunk_index"): doc_item
            for doc_item in doc_cache[doc_id]
        }
        base_score = item.get("score", 0.0) * 0.92

        for neighbor_idx in range(max(0, chunk_index - window), chunk_index + window + 1):
            neighbor = neighbors_by_index.get(neighbor_idx)
            if not neighbor or neighbor["id"] in seen_ids:
                continue
            enriched = dict(neighbor)
            enriched["score"] = base_score
            expanded.append(enriched)
            seen_ids.add(neighbor["id"])

    return expanded


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


# â”€â”€ Legal parent expansion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Query expansion & HyDE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


# â”€â”€ Semantic retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ Determine queries to run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    queries = [question]
    if getattr(settings, "query_expansion_enabled", False):
        with metrics.time_block("chat.query_expansion"):
            queries = _expand_query_llm(question, settings, request_id)

    search_query = normalize_query_numerals(question)

    # â”€â”€ Vector search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Convert distance â†’ score (cosine distance 0-2 â†’ score 0-1) â”€â”€â”€â”€â”€
    for hit in all_hits:
        if "score" not in hit and "distance" in hit:
            hit["score"] = max(0.0, 1.0 - hit["distance"])

    # â”€â”€ Deduplicate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Cross-encoder reranking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Structural reranking + threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
                    "citation_label": "consulta analitica executada",
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


# â”€â”€ Query intent classification (Gap 4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class QueryIntent(Enum):
    COUNT_STRUCTURAL = "count_structural"  # "quantos capitulos tem?"
    LIST_STRUCTURAL = "list_structural"  # "quais secoes existem no capitulo v?"
    LOCATE_STRUCTURAL = "locate_structural"  # "qual e o ultimo capitulo?"
    CONTAINS_STRUCTURAL = "contains_structural"  # "o capitulo v tem secao iii?"
    SUMMARY_STRUCTURAL = "summary_structural"  # "resuma o capÃ­tulo 2"
    QUESTION_STRUCTURAL = "question_structural"  # "o que diz o capÃ­tulo 2?"
    QUESTION_FACTUAL = "question_factual"  # "Ã© proibido remunerar?"
    LOCATE_EXCERPT = "locate_excerpt"  # "art. 41"
    COMPARISON = "comparison"  # "compare capÃ­tulo 1 e 2"

_SUMMARY_PATTERNS = [
    re.compile(r"\b(?:resum[aeiou]|sintetiz[ae]|expliq[ue]|descrev[ae]|vis[aÃ£]o\s+(?:geral|executiva))\b", re.IGNORECASE),
]
_STRUCTURAL_REF_PATTERNS = [
    re.compile(r"\b(?:cap[iÃ­]tulo|se[cÃ§][aÃ£]o|t[iÃ­]tulo|parte)\s+[IVXLCDM\d]+", re.IGNORECASE),
]
_COMPARISON_PATTERNS = [
    re.compile(r"\bcompar[ae]\b", re.IGNORECASE),
    re.compile(r"\bdiferenÃ§a\s+entre\b", re.IGNORECASE),
    re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE),
]


def classify_query_intent(question: str) -> QueryIntent:
    """Classify user question into an intent category.

    Uses regex (fast) or embeddings (more accurate) based on config.
    """
    settings = get_settings()
    if getattr(settings, "intent_classifier", "regex") == "embeddings":
        return _classify_intent_embeddings(question)
    return _classify_intent_regex(question)


def _classify_intent_regex(question: str) -> QueryIntent:
    """Regex-based intent classification (fast, default).

    Priority: summary_structural > comparison > question_structural > locate_excerpt > factual
    """
    q = question.strip()
    q_norm = _normalize_for_match(q)

    has_summary = bool(
        re.search(r"\b(resum|sintetiz|explique|descrev|visao\s+(geral|executiva))", q_norm)
    )
    has_count = bool(
        re.search(r"\b(quantos|quantas|numero\s+de|qtd\.?)\b", q_norm)
    )
    has_list = bool(
        re.search(r"\b(quais|liste|listar|lista|relacione)\b", q_norm)
    )
    has_locate = bool(
        re.search(r"\b(onde\s+esta|onde\s+fica|ultimo|ultima|primeiro|primeira)\b", q_norm)
    )
    has_contains = bool(
        re.search(r"\b(tem|possui|contem|inclui|ha|existe|existem)\b", q_norm)
    )
    has_structural_ref = bool(
        re.search(r"\b(capitulo|secao|titulo|parte)\s+([ivxlcdm]+|\d{1,4})\b", q_norm)
    )
    has_structural_noun = bool(
        re.search(r"\b(capitulo|capitulos|secao|secoes|titulo|titulos|artigo|artigos)\b", q_norm)
    )
    has_comparison = bool(
        re.search(r"\b(compar\w*|diferenca\s+entre|versus|vs\.?)\b", q_norm)
    )

    if has_summary and has_structural_ref:
        return QueryIntent.SUMMARY_STRUCTURAL
    if has_count and has_structural_noun:
        return QueryIntent.COUNT_STRUCTURAL
    if has_list and has_structural_noun:
        return QueryIntent.LIST_STRUCTURAL
    if has_locate and has_structural_noun:
        return QueryIntent.LOCATE_STRUCTURAL
    if has_contains and len(_extract_structural_targets(q)) >= 2:
        return QueryIntent.CONTAINS_STRUCTURAL
    if has_comparison and has_structural_ref:
        return QueryIntent.COMPARISON
    if has_structural_ref:
        # Check if it's a locate request (short query with just article/chapter ref)
        if _is_structural_query(q) and len(q.split()) <= 6:
            return QueryIntent.LOCATE_EXCERPT
        return QueryIntent.QUESTION_STRUCTURAL
    return QueryIntent.QUESTION_FACTUAL


# â”€â”€ Embeddings-based intent classifier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Exemplar queries per intent â€” used as few-shot anchors for similarity
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
        "Resuma o capÃ­tulo II",
        "Sintetize a seÃ§Ã£o III",
        "VisÃ£o geral do tÃ­tulo I",
        "Explique o capÃ­tulo 3 em linguagem simples",
        "FaÃ§a um resumo executivo do capÃ­tulo IV",
    ],
    QueryIntent.QUESTION_STRUCTURAL: [
        "O que diz o capÃ­tulo II sobre obrigaÃ§Ãµes?",
        "Quais sÃ£o os direitos previstos na seÃ§Ã£o I?",
        "O capÃ­tulo 5 trata de quÃª?",
        "Que artigos estÃ£o no tÃ­tulo III?",
    ],
    QueryIntent.QUESTION_FACTUAL: [
        "Qual o prazo para pagamento de rescisÃ£o?",
        "Ã‰ proibido remunerar diretores?",
        "Quantos membros tem o conselho?",
        "Qual Ã© o quÃ³rum de deliberaÃ§Ã£o?",
    ],
    QueryIntent.LOCATE_EXCERPT: [
        "Art. 41",
        "Artigo 15",
        "CapÃ­tulo II",
        "SeÃ§Ã£o III",
    ],
    QueryIntent.COMPARISON: [
        "Compare o capÃ­tulo 1 e o capÃ­tulo 2",
        "Qual a diferenÃ§a entre a seÃ§Ã£o I e a seÃ§Ã£o II?",
        "CapÃ­tulo III versus capÃ­tulo IV",
    ],
}

# Cache for pre-computed exemplar embeddings (populated on first call)
_intent_exemplar_embeddings: dict[QueryIntent, list[list[float]]] | None = None


def _get_intent_exemplar_embeddings() -> dict[QueryIntent, list[list[float]]]:
    """Compute and cache exemplar embeddings for each intent."""
    global _intent_exemplar_embeddings
    if _intent_exemplar_embeddings is not None:
        return _intent_exemplar_embeddings

    _intent_exemplar_embeddings = {}
    for intent, exemplars in _INTENT_EXEMPLARS.items():
        _intent_exemplar_embeddings[intent] = llm.embed(exemplars)

    log_event(logger, 20, "Intent exemplar embeddings computed",
              intents=len(_intent_exemplar_embeddings))
    return _intent_exemplar_embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _classify_intent_embeddings(question: str) -> QueryIntent:
    """ML-based intent classification using embedding similarity.

    Embeds the question and compares against pre-computed exemplar embeddings
    for each intent. Returns the intent with the highest average similarity.
    """
    exemplar_embs = _get_intent_exemplar_embeddings()

    # Embed the query
    q_embedding = llm.embed([question.strip()])[0]

    best_intent = QueryIntent.QUESTION_FACTUAL
    best_score = -1.0

    for intent, emb_list in exemplar_embs.items():
        # Average similarity across all exemplars for this intent
        scores = [_cosine_similarity(q_embedding, e) for e in emb_list]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        if avg_score > best_score:
            best_score = avg_score
            best_intent = intent

    log_event(logger, 10, "Intent classified via embeddings",
              question=question[:80], intent=best_intent.value, score=round(best_score, 4))
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


# â”€â”€ Main answer function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ Guardrails â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    question = sanitize_question(question)
    history = sanitize_history(history or [])

    injection = detect_injection(question)
    if injection:
        log_event(logger, 30, "Prompt injection detected in question", pattern=injection, request_id=request_id)
        return ChatResult(
            answer="NÃ£o foi possÃ­vel processar essa pergunta. Reformule sua solicitaÃ§Ã£o.",
            sources=[],
            request_id=request_id,
        )

    # â”€â”€ Intent classification + summary shortcut â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
            answer="NÃ£o encontrei informaÃ§Ãµes relevantes nos documentos disponÃ­veis para essa coleÃ§Ã£o.",
            sources=[],
            request_id=request_id,
        )

    # â”€â”€ Parent expansion for legal chunks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Sanitize + compress + trim + format â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ LLM generation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


# â”€â”€ Streaming â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class StreamContext:
    """Holds retrieval results for streaming â€” sources are sent before the LLM tokens."""
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
            answer="NÃ£o foi possÃ­vel processar essa pergunta. Reformule sua solicitaÃ§Ã£o.",
            sources=[],
            request_id=request_id,
        )

    # â”€â”€ Intent classification + summary shortcut for streaming â”€â”€â”€â”€
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
            answer="NÃ£o encontrei informaÃ§Ãµes relevantes nos documentos disponÃ­veis para essa coleÃ§Ã£o.",
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

