"""Semantic retrieval (vector search) and RAG generation."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from src import llm, vectordb
from src.config import get_settings
from src.guardrails import detect_injection, sanitize_context_chunk, sanitize_history, sanitize_question
from src.lexical import normalize_query_numerals
from src.observability import metrics
from src.prompts import build_rag_messages, format_context, get_rag_system
from src.utils import get_logger, log_event

logger = get_logger(__name__)


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


def _extract_chapter_refs(question: str) -> set[str]:
    expanded = _normalize_for_match(normalize_query_numerals(question))
    refs: set[str] = set()
    for match in _CHAPTER_REF_RE.finditer(expanded):
        refs.add(match.group(1).lower())
    return refs


def _extract_chapter_refs_from_text(text: str) -> set[str]:
    normalized = _normalize_for_match(normalize_query_numerals(text))
    refs: set[str] = set()
    for match in _CHAPTER_REF_RE.finditer(normalized):
        refs.add(match.group(1).lower())
    return refs


def _expand_refs_arabic_roman(refs: set[str]) -> set[str]:
    """Expand chapter refs to include both Arabic and Roman numeral forms.

    E.g., {"2"} → {"2", "ii"}, {"ii"} → {"ii", "2"}, {"iv"} → {"iv", "4"}
    """
    from src.lexical import _int_to_roman, _roman_to_int
    expanded = set(refs)
    for ref in refs:
        if ref.isdigit():
            n = int(ref)
            if 1 <= n <= 3999:
                expanded.add(_int_to_roman(n).lower())
        elif ref.isalpha():
            n = _roman_to_int(ref.upper())
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
    """Keep only chunks that match the exact requested chapter/section reference."""
    refs = _extract_chapter_refs(question)
    if not refs:
        return results, False
    matched = [item for item in results if _item_matches_chapter_refs(item, refs)]
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
    """Resolve chapter scope directly from chunk metadata for top seeded documents."""
    refs = _extract_chapter_refs(question)
    if not refs or not seed_results:
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
            if not _item_matches_chapter_refs(item, refs):
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

class QueryIntent(Enum):
    SUMMARY_STRUCTURAL = "summary_structural"  # "resuma o capítulo 2"
    QUESTION_STRUCTURAL = "question_structural"  # "o que diz o capítulo 2?"
    QUESTION_FACTUAL = "question_factual"  # "é proibido remunerar?"
    LOCATE_EXCERPT = "locate_excerpt"  # "art. 41"
    COMPARISON = "comparison"  # "compare capítulo 1 e 2"

_SUMMARY_PATTERNS = [
    re.compile(r"\b(?:resum[aeiou]|sintetiz[ae]|expliq[ue]|descrev[ae]|vis[aã]o\s+(?:geral|executiva))\b", re.IGNORECASE),
]
_STRUCTURAL_REF_PATTERNS = [
    re.compile(r"\b(?:cap[ií]tulo|se[cç][aã]o|t[ií]tulo|parte)\s+[IVXLCDM\d]+", re.IGNORECASE),
]
_COMPARISON_PATTERNS = [
    re.compile(r"\bcompar[ae]\b", re.IGNORECASE),
    re.compile(r"\bdiferença\s+entre\b", re.IGNORECASE),
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
    has_summary = any(p.search(q) for p in _SUMMARY_PATTERNS)
    has_structural_ref = any(p.search(q) for p in _STRUCTURAL_REF_PATTERNS)
    has_comparison = any(p.search(q) for p in _COMPARISON_PATTERNS)

    if has_summary and has_structural_ref:
        return QueryIntent.SUMMARY_STRUCTURAL
    if has_comparison and has_structural_ref:
        return QueryIntent.COMPARISON
    if has_structural_ref:
        # Check if it's a locate request (short query with just article/chapter ref)
        if _is_structural_query(q) and len(q.split()) <= 6:
            return QueryIntent.LOCATE_EXCERPT
        return QueryIntent.QUESTION_STRUCTURAL
    return QueryIntent.QUESTION_FACTUAL


# ── Embeddings-based intent classifier ─────────────────────────────────────

# Exemplar queries per intent — used as few-shot anchors for similarity
_INTENT_EXEMPLARS: dict[QueryIntent, list[str]] = {
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
    """Try to resolve a summary_structural query from pre-computed summaries.

    Returns a summary dict if found, None otherwise.
    """
    from src import controlplane

    # Extract chapter/section reference from question
    refs = _extract_chapter_refs(question)
    if not refs:
        return None

    # Build search variants: Arabic + Roman forms of each ref
    search_terms = set()
    for ref in refs:
        search_terms.add(ref)
        search_terms.add(ref.upper())
        # Arabic → Roman
        if ref.isdigit():
            from src.lexical import _int_to_roman
            roman = _int_to_roman(int(ref))
            search_terms.add(roman)
        # Roman → Arabic
        elif ref.upper().isalpha():
            from src.lexical import _roman_to_int
            arabic = _roman_to_int(ref.upper())
            if arabic > 0:
                search_terms.add(str(arabic))

    # Try to find matching summaries
    for term in search_terms:
        for node_type in ("capitulo", "secao", "titulo"):
            summaries = controlplane.find_summaries_by_label(
                workspace_id=workspace_id,
                collection=collection,
                label_query=term,
                node_type=node_type,
            )
            if summaries:
                return summaries[0]

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
    )
    return ChatResult(answer=answer_text, sources=sources, request_id=request_id)


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

    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        summary = _resolve_structural_summary(question, collection, workspace_id)
        if summary:
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
                answer="Nao localizei trechos suficientes do capitulo solicitado nos documentos desta colecao.",
                sources=[],
                request_id=request_id,
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

    if intent == QueryIntent.SUMMARY_STRUCTURAL:
        summary = _resolve_structural_summary(question, collection, workspace_id)
        if summary:
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
                answer="Nao localizei trechos suficientes do capitulo solicitado nos documentos desta colecao.",
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
