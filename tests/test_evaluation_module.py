import json
import math
from pathlib import Path

from src.evaluation import (
    average_precision,
    evaluate_ab,
    evaluate_retrieval_snapshot,
    load_gold_dataset,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    GOLD_CORPUS,
    GOLD_QUERIES,
)


# --- existing tests ---


def test_metric_helpers_work():
    assert precision_at_k(["a", "b"], ["a"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["a"], 2) == 1.0


def test_evaluation_snapshot_has_expected_shape():
    snapshot = evaluate_retrieval_snapshot(top_k=3)

    assert snapshot["dataset"] == "embedded_gold_ptbr"
    assert snapshot["queries"] > 0
    assert snapshot["vector"]["precision_at_k"] >= 0.0
    assert snapshot["vector"]["recall_at_k"] >= 0.0


# --- NDCG tests ---


def test_ndcg_perfect_ranking():
    """Perfect ranking should yield NDCG = 1.0."""
    retrieved = ["d1", "d2", "d3"]
    grades = {"d1": 2, "d2": 1, "d3": 0}
    assert ndcg_at_k(retrieved, grades, 3) == 1.0


def test_ndcg_reversed_ranking():
    """Reversed ranking should yield NDCG < 1.0."""
    retrieved = ["d3", "d2", "d1"]
    grades = {"d1": 2, "d2": 1, "d3": 0}
    score = ndcg_at_k(retrieved, grades, 3)
    assert 0.0 < score < 1.0


def test_ndcg_empty_inputs():
    assert ndcg_at_k([], {"d1": 2}, 5) == 0.0
    assert ndcg_at_k(["d1"], {}, 5) == 0.0


def test_ndcg_single_relevant():
    """Single relevant doc at rank 1 should give NDCG = 1.0."""
    assert ndcg_at_k(["d1"], {"d1": 2}, 1) == 1.0


def test_ndcg_respects_k_cutoff():
    retrieved = ["d2", "d1"]  # d1 is most relevant but at rank 2
    grades = {"d1": 2, "d2": 0}
    # With k=1, only d2 is considered (grade 0), NDCG should be 0
    assert ndcg_at_k(retrieved, grades, 1) == 0.0


# --- MRR tests ---


def test_mrr_first_position():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_position():
    assert mrr(["x", "a", "c"], ["a"]) == 0.5


def test_mrr_not_found():
    assert mrr(["x", "y", "z"], ["a"]) == 0.0


def test_mrr_multiple_relevant():
    """MRR considers only the FIRST relevant hit."""
    assert mrr(["x", "a", "b"], ["a", "b"]) == 0.5


# --- Average Precision tests ---


def test_avg_precision_perfect():
    assert average_precision(["a", "b"], ["a", "b"]) == 1.0


def test_avg_precision_no_hits():
    assert average_precision(["x", "y"], ["a"]) == 0.0


def test_avg_precision_partial():
    # Retrieved: [x, a] — relevant: [a]
    # Hit at position 2: precision = 1/2
    # AP = 0.5 / 1 = 0.5
    assert average_precision(["x", "a"], ["a"]) == 0.5


def test_avg_precision_interleaved():
    # Retrieved: [a, x, b] — relevant: [a, b]
    # Hit at pos 1: precision = 1/1 = 1.0
    # Hit at pos 3: precision = 2/3 ≈ 0.6667
    # AP = (1.0 + 2/3) / 2 = 0.8333...
    ap = average_precision(["a", "x", "b"], ["a", "b"])
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-9


def test_avg_precision_empty_relevant():
    assert average_precision(["a", "b"], []) == 1.0


# --- Snapshot with expanded metrics ---


def test_snapshot_includes_all_metrics():
    snapshot = evaluate_retrieval_snapshot(top_k=5)
    expected_keys = {"precision_at_k", "recall_at_k", "ndcg_at_k", "mrr", "avg_precision"}
    assert set(snapshot["vector"].keys()) == expected_keys


def test_snapshot_with_external_dataset(tmp_path):
    dataset = {
        "corpus": [
            {"id": "d1", "text": "Ferias de 30 dias apos 12 meses de trabalho."},
            {"id": "d2", "text": "Aviso previo de 30 dias minimo."},
        ],
        "queries": [
            {"question": "Quantos dias de ferias?", "relevant": ["d1"], "partially_relevant": []},
        ],
    }
    path = tmp_path / "test_gold.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    snapshot = evaluate_retrieval_snapshot(top_k=2, dataset_path=str(path))
    assert snapshot["queries"] == 1
    assert snapshot["dataset"] == str(path)


# --- load_gold_dataset ---


def test_load_gold_dataset_fallback():
    """Without file, falls back to hardcoded constants."""
    corpus, queries = load_gold_dataset(path="/nonexistent/path.json")
    assert corpus == GOLD_CORPUS
    assert queries == GOLD_QUERIES


def test_load_gold_dataset_from_file(tmp_path):
    data = {
        "corpus": [{"id": "x1", "text": "test"}],
        "queries": [{"question": "q?", "relevant": ["x1"]}],
    }
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    corpus, queries = load_gold_dataset(path)
    assert len(corpus) == 1
    assert corpus[0]["id"] == "x1"


# --- evaluate_ab ---


def test_evaluate_ab_basic():
    def strategy_perfect(question, corpus, ids, top_k):
        return ids[:top_k]

    def strategy_reverse(question, corpus, ids, top_k):
        return list(reversed(ids))[:top_k]

    result = evaluate_ab(strategy_perfect, strategy_reverse, top_k=3)
    assert result["queries"] > 0
    assert "strategy_a" in result
    assert "strategy_b" in result
    assert "deltas_b_minus_a" in result
    assert "per_query_wins" in result


def test_evaluate_ab_identical_strategies():
    def strategy(question, corpus, ids, top_k):
        return ids[:top_k]

    result = evaluate_ab(strategy, strategy, top_k=3)
    # Deltas should be zero
    for key, val in result["deltas_b_minus_a"].items():
        assert val == 0.0, f"Delta for {key} should be 0, got {val}"
    # All ties
    for key, wins in result["per_query_wins"].items():
        assert wins["a_wins"] == 0
        assert wins["b_wins"] == 0
