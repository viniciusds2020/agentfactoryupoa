"""Structural matching functions for legal document retrieval."""
from __future__ import annotations

import re
import unicodedata

from src import vectordb
from src.lexical import normalize_query_numerals


# ── Normalize / tokenize helpers ─────────────────────────────────────────────

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


# ── Regex patterns ───────────────────────────────────────────────────────────

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


# ── Ref extraction ───────────────────────────────────────────────────────────

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

    E.g., {"2"} -> {"2", "ii"}, {"ii"} -> {"ii", "2"}, {"iv"} -> {"iv", "4"}
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


# ── Item matching ────────────────────────────────────────────────────────────

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
        # (e.g., "CAPITULO"), but numerals still survive.
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


# ── Context expansion ────────────────────────────────────────────────────────

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
