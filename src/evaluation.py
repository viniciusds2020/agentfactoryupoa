"""Offline retrieval evaluation helpers for observability endpoints."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

from src.lexical import tokenize


GOLD_CORPUS = [
    {"id": "c1", "text": "O prazo para pagamento de rescisao e de 10 dias uteis conforme Art. 477 da CLT."},
    {"id": "c2", "text": "Ferias proporcionais devem ser pagas na rescisao do contrato de trabalho."},
    {"id": "c3", "text": "O FGTS deve ser depositado ate o dia 7 do mes seguinte ao trabalhado."},
    {"id": "c4", "text": "O aviso previo pode ser trabalhado ou indenizado, conforme escolha do empregador."},
    {"id": "c5", "text": "A jornada de trabalho padrao e de 8 horas diarias e 44 horas semanais."},
    {"id": "c6", "text": "O salario minimo e reajustado anualmente pelo governo federal."},
    {"id": "c7", "text": "O contrato de experiencia tem prazo maximo de 90 dias."},
    {"id": "c8", "text": "O decimo terceiro salario e pago em duas parcelas anuais."},
    {"id": "c9", "text": "A licenca-maternidade tem duracao de 120 dias, podendo ser estendida para 180 dias."},
    {"id": "c10", "text": "Horas extras devem ser remuneradas com adicional minimo de 50% sobre a hora normal."},
]

GOLD_QUERIES = [
    {"question": "Qual o prazo para pagar rescisao?", "relevant": ["c1"]},
    {"question": "Como funciona o deposito do FGTS?", "relevant": ["c3"]},
    {"question": "Quanto tempo dura o contrato de experiencia?", "relevant": ["c7"]},
    {"question": "Quando e pago o decimo terceiro?", "relevant": ["c8"]},
    {"question": "Qual o adicional de hora extra?", "relevant": ["c10"]},
]


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & set(relevant_ids)) / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not relevant_ids:
        return 1.0
    return len(set(top_k) & set(relevant_ids)) / len(relevant_ids)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs.
        relevance_grades: Mapping of doc_id -> relevance grade (0, 1, 2).
        k: Cutoff position.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    top_k = retrieved_ids[:k]
    if not top_k or not relevance_grades:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = relevance_grades.get(doc_id, 0)
        dcg += rel / math.log2(i + 2)

    ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank -- reciprocal of the rank of the first relevant result."""
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Average Precision for a single query."""
    if not relevant_ids:
        return 1.0
    relevant_set = set(relevant_ids)
    hits = 0
    sum_precision = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    if hits == 0:
        return 0.0
    return sum_precision / len(relevant_ids)


def _pseudo_vector_results(question: str, top_k: int, corpus: list[dict] | None = None) -> list[dict]:
    """Simulate vector search using token overlap scoring (offline evaluation)."""
    corpus = corpus or GOLD_CORPUS
    q_tokens = set(tokenize(question))
    scored: list[tuple[float, dict]] = []
    for item in corpus:
        text_tokens = set(tokenize(item["text"]))
        score = len(q_tokens & text_tokens) / max(len(text_tokens), 1)
        scored.append((score, {"id": item["id"], "text": item["text"], "metadata": {"doc_id": item["id"]}, "score": score}))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for score, item in scored[:top_k] if score > 0]


def load_gold_dataset(path: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    """Load gold dataset from JSON file, falling back to hardcoded constants.

    Expected JSON structure::

        {
            "corpus": [{"id": "c1", "text": "..."}],
            "queries": [
                {
                    "question": "...",
                    "relevant": ["c1"],
                    "partially_relevant": ["c2"],
                    "query_type": "keyword"
                }
            ]
        }

    Returns:
        Tuple of (corpus, queries).
    """
    if path is not None:
        p = Path(path)
    else:
        p = Path("data/eval/gold_dataset.json")

    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return data["corpus"], data["queries"]
    return GOLD_CORPUS, GOLD_QUERIES


def _build_relevance_grades(query: dict) -> dict[str, int]:
    """Build a relevance grade map from a query entry.

    ``relevant`` IDs get grade 2, ``partially_relevant`` get grade 1.
    """
    grades: dict[str, int] = {}
    for doc_id in query.get("partially_relevant", []):
        grades[doc_id] = 1
    for doc_id in query["relevant"]:
        grades[doc_id] = 2
    return grades


def _compute_all_metrics(
    retrieved_ids: list[str],
    query: dict,
    k: int,
) -> dict[str, float]:
    """Compute all metrics for a single query."""
    relevant = query["relevant"]
    grades = _build_relevance_grades(query)
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
        "ndcg_at_k": ndcg_at_k(retrieved_ids, grades, k),
        "mrr": mrr(retrieved_ids, relevant),
        "avg_precision": average_precision(retrieved_ids, relevant),
    }


def _avg_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Average metrics across multiple queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    total = len(per_query)
    return {key: round(sum(q[key] for q in per_query) / total, 4) for key in keys}


def evaluate_retrieval_snapshot(
    top_k: int = 5,
    dataset_path: str | Path | None = None,
) -> dict[str, object]:
    """Run evaluation on vector search using gold dataset.

    Computes precision, recall, NDCG, MRR and MAP.
    """
    corpus, queries = load_gold_dataset(dataset_path)
    vector_per_query: list[dict[str, float]] = []

    for query in queries:
        vector_hits = _pseudo_vector_results(query["question"], top_k=top_k, corpus=corpus)
        vector_ids = [item["id"] for item in vector_hits]
        vector_per_query.append(_compute_all_metrics(vector_ids, query, top_k))

    return {
        "dataset": "embedded_gold_ptbr" if dataset_path is None else str(dataset_path),
        "queries": len(queries),
        "top_k": top_k,
        "vector": _avg_metrics(vector_per_query),
    }


# ── Structural summarization evaluation (Gap 5) ─────────────────────────────

STRUCTURAL_GOLD_QUERIES = [
    {
        "question": "Resuma o Capítulo II",
        "target_type": "capitulo",
        "target_numeral": "II",
        "expected_articles": [],
        "query_intent": "summary_structural",
    },
    {
        "question": "Quais as vedações do Capítulo X?",
        "target_type": "capitulo",
        "target_numeral": "X",
        "expected_articles": [],
        "query_intent": "summary_structural",
    },
    {
        "question": "Explique a Seção III em linguagem simples",
        "target_type": "secao",
        "target_numeral": "III",
        "expected_articles": [],
        "query_intent": "summary_structural",
    },
    {
        "question": "Liste os artigos cobertos pelo Capítulo 2",
        "target_type": "capitulo",
        "target_numeral": "2",
        "expected_articles": [],
        "query_intent": "summary_structural",
    },
    {
        "question": "Qual a visão executiva do Capítulo I?",
        "target_type": "capitulo",
        "target_numeral": "I",
        "expected_articles": [],
        "query_intent": "summary_structural",
    },
]


def structural_hit_at_1(
    retrieved_node_type: str,
    retrieved_numeral: str,
    target_type: str,
    target_numeral: str,
) -> float:
    """Check if the retrieved node matches the target (type + numeral). Returns 1.0 or 0.0."""
    if retrieved_node_type.lower() == target_type.lower():
        if _numerals_equivalent(retrieved_numeral, target_numeral):
            return 1.0
    return 0.0


def _normalize_numeral_token(value: str) -> str:
    from src.lexical import _roman_to_int
    token = value.strip().upper()
    if not token:
        return ""
    if token.isdigit():
        return str(int(token))
    arabic = _roman_to_int(token)
    if arabic > 0:
        return str(arabic)
    return token


def _numerals_equivalent(a: str, b: str) -> bool:
    na = _normalize_numeral_token(a)
    nb = _normalize_numeral_token(b)
    if na and nb:
        return na == nb
    return a.strip().upper() == b.strip().upper()


def article_coverage(
    summary_articles: list[str],
    expected_articles: list[str],
) -> float:
    """Fraction of expected articles present in the summary."""
    if not expected_articles:
        return 1.0  # If no expectation, coverage is trivially complete
    expected_set = set(a.strip().lower() for a in expected_articles)
    covered = set(a.strip().lower() for a in summary_articles)
    return len(expected_set & covered) / len(expected_set)


def citation_faithfulness(
    answer_text: str,
    node_label: str,
) -> float:
    """Check if citations in the answer reference the correct structural node.

    Returns fraction of citation markers [N] that appear in a context related
    to the target label. Simple heuristic: checks if the node label appears
    in the answer text.
    """
    import re as _re
    citations = _re.findall(r"\[\d+\]", answer_text)
    if not citations:
        return 0.0
    # Heuristic: if the label is mentioned, citations are likely faithful
    label_norm = node_label.strip().lower()
    answer_norm = answer_text.lower()
    if label_norm in answer_norm:
        return 1.0
    # Check if any significant word from the label appears in the answer
    parts = [p for p in label_norm.split() if len(p) > 3]
    if parts and any(part in answer_norm for part in parts):
        return 1.0
    return 0.0


def section_boundary_precision(
    answer_text: str,
    target_label: str,
    other_labels: list[str],
) -> float:
    """Check that the answer doesn't mix content from other chapters/sections.

    Returns 1.0 if no other labels are mentioned, or a fraction based on
    how many extraneous labels appear.
    """
    if not other_labels:
        return 1.0
    answer_norm = answer_text.lower()
    contamination = 0
    for label in other_labels:
        if label.strip().lower() in answer_norm:
            contamination += 1
    return 1.0 - (contamination / len(other_labels))


def evaluate_structural_summary(
    summaries: list[dict],
    dataset: list[dict] | None = None,
) -> dict:
    """Evaluate structural summarization quality.

    Args:
        summaries: Available pre-computed summaries as list of dicts
            with keys: node_type, numeral, label, artigos_cobertos, resumo_executivo
        dataset: Gold queries to evaluate against. Defaults to STRUCTURAL_GOLD_QUERIES.

    Returns:
        Dict with per-metric averages and per-query details.
    """
    queries = dataset or STRUCTURAL_GOLD_QUERIES

    results: list[dict] = []
    found_count = 0
    valid_count = 0
    fallback_count = 0
    fallback_success = 0
    for query in queries:
        target_type = query["target_type"]
        target_numeral = query["target_numeral"]

        # Try to find matching summary
        matched = None
        for s in summaries:
            if s.get("node_type", "").lower() != target_type.lower():
                continue
            numeral_in_summary = str(s.get("numeral", "")).strip()
            if not numeral_in_summary:
                m = re.search(r"(?:T[IÍ]TULO|CAP[IÍ]TULO|SE[CÇ][AÃ]O)\s+([IVXLCDM]+|\d+)", s.get("label", ""), re.IGNORECASE)
                numeral_in_summary = m.group(1) if m else ""
            if _numerals_equivalent(numeral_in_summary, target_numeral):
                matched = s
                break

        if matched:
            found_count += 1
            status = str(matched.get("status", "generated")).lower()
            if status != "invalid":
                valid_count += 1
            if status == "fallback_only":
                fallback_count += 1
                fallback_success += 1
            hit = structural_hit_at_1(
                matched.get("node_type", ""),
                numeral_in_summary,
                target_type,
                target_numeral,
            )
            coverage = article_coverage(
                matched.get("artigos_cobertos", []),
                query.get("expected_articles", []),
            )
            scope_acc = 1.0 if hit == 1.0 else 0.0
        else:
            hit = 0.0
            coverage = 0.0
            scope_acc = 0.0

        results.append({
            "question": query["question"],
            "target": f"{target_type} {target_numeral}",
            "matched": matched.get("label", "") if matched else "",
            "status": matched.get("status", "") if matched else "",
            "structural_hit_at_1": hit,
            "summary_scope_accuracy": scope_acc,
            "article_coverage": coverage,
            "citation_scope_precision": scope_acc,  # offline proxy
        })

    # Averages
    n = len(results) or 1
    avg_hit = sum(r["structural_hit_at_1"] for r in results) / n
    avg_coverage = sum(r["article_coverage"] for r in results) / n
    avg_scope = sum(r["summary_scope_accuracy"] for r in results) / n
    avg_citation_scope = sum(r["citation_scope_precision"] for r in results) / n
    summary_found_rate = found_count / n
    summary_valid_rate = (valid_count / found_count) if found_count else 0.0
    fallback_success_rate = (fallback_success / fallback_count) if fallback_count else 0.0

    return {
        "structural_hit_at_1": round(avg_hit, 4),
        "article_coverage": round(avg_coverage, 4),
        "summary_found_rate": round(summary_found_rate, 4),
        "summary_valid_rate": round(summary_valid_rate, 4),
        "summary_scope_accuracy": round(avg_scope, 4),
        "citation_scope_precision": round(avg_citation_scope, 4),
        "fallback_success_rate": round(fallback_success_rate, 4),
        "queries_evaluated": len(results),
        "per_query": results,
    }


def evaluate_ab(
    strategy_a: Callable[[str, list[dict], list[str], int], list[str]],
    strategy_b: Callable[[str, list[dict], list[str], int], list[str]],
    top_k: int = 5,
    dataset_path: str | Path | None = None,
) -> dict[str, object]:
    """Compare two retrieval strategies side-by-side on the gold dataset.

    Each strategy is a callable: ``(question, corpus, ids, top_k) -> list[str]``
    returning an ordered list of retrieved document IDs.

    Returns per-metric averages for both strategies plus deltas and per-query wins.
    """
    corpus, queries = load_gold_dataset(dataset_path)
    ids = [item["id"] for item in corpus]

    a_per_query: list[dict[str, float]] = []
    b_per_query: list[dict[str, float]] = []

    for query in queries:
        a_ids = strategy_a(query["question"], corpus, ids, top_k)
        b_ids = strategy_b(query["question"], corpus, ids, top_k)
        a_per_query.append(_compute_all_metrics(a_ids, query, top_k))
        b_per_query.append(_compute_all_metrics(b_ids, query, top_k))

    a_avg = _avg_metrics(a_per_query)
    b_avg = _avg_metrics(b_per_query)

    deltas = {key: round(b_avg[key] - a_avg[key], 4) for key in a_avg}

    wins: dict[str, dict[str, int]] = {}
    for key in a_avg:
        a_wins = sum(1 for a, b in zip(a_per_query, b_per_query) if a[key] > b[key])
        b_wins = sum(1 for a, b in zip(a_per_query, b_per_query) if b[key] > a[key])
        ties = len(queries) - a_wins - b_wins
        wins[key] = {"a_wins": a_wins, "b_wins": b_wins, "ties": ties}

    return {
        "queries": len(queries),
        "top_k": top_k,
        "strategy_a": a_avg,
        "strategy_b": b_avg,
        "deltas_b_minus_a": deltas,
        "per_query_wins": wins,
    }
