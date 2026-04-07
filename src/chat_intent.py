"""Query intent classification for chat routing."""
from __future__ import annotations

import re
from enum import Enum

from src import llm
from src.config import get_settings
from src.chat_structural import (
    _extract_structural_targets,
    _is_structural_query,
    _normalize_for_match,
)
from src.utils import get_logger, log_event

logger = get_logger(__name__)


# ── Query intent classification ──────────────────────────────────────────────

class QueryIntent(Enum):
    COUNT_STRUCTURAL = "count_structural"  # "quantos capitulos tem?"
    LIST_STRUCTURAL = "list_structural"  # "quais secoes existem no capitulo v?"
    LOCATE_STRUCTURAL = "locate_structural"  # "qual e o ultimo capitulo?"
    CONTAINS_STRUCTURAL = "contains_structural"  # "o capitulo v tem secao iii?"
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


# ── Embeddings-based intent classifier ─────────────────────────────────────

# Exemplar queries per intent -- used as few-shot anchors for similarity
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
