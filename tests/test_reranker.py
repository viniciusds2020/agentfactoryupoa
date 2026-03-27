"""Tests for cross-encoder reranking module."""
from unittest.mock import MagicMock, patch

from src.reranker import rerank


def _make_candidates(n: int) -> list[dict]:
    return [
        {"id": f"d{i}", "text": f"Document text number {i}", "score": 0.5 - i * 0.01}
        for i in range(n)
    ]


@patch("src.reranker._load_cross_encoder")
def test_rerank_reorders_by_cross_encoder_score(mock_load):
    model = MagicMock()
    # Reverse the order: last candidate gets highest score
    model.predict.return_value = [0.1, 0.3, 0.9, 0.5]
    mock_load.return_value = model

    candidates = _make_candidates(4)
    result = rerank("test query", candidates, model_name="test-model", top_k=4)

    assert len(result) == 4
    assert result[0]["id"] == "d2"  # score 0.9
    assert result[1]["id"] == "d3"  # score 0.5
    assert result[2]["id"] == "d1"  # score 0.3
    assert result[3]["id"] == "d0"  # score 0.1


@patch("src.reranker._load_cross_encoder")
def test_rerank_respects_top_k(mock_load):
    model = MagicMock()
    model.predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
    mock_load.return_value = model

    candidates = _make_candidates(5)
    result = rerank("test query", candidates, top_k=3)

    assert len(result) == 3


@patch("src.reranker._load_cross_encoder")
def test_rerank_stores_ce_score(mock_load):
    model = MagicMock()
    model.predict.return_value = [0.42]
    mock_load.return_value = model

    candidates = [{"id": "d1", "text": "some text", "score": 0.1}]
    result = rerank("query", candidates, top_k=1)

    assert result[0]["ce_score"] == 0.42
    assert result[0]["base_score"] == 0.1
    assert 0.0 <= result[0]["ce_score_norm"] <= 1.0
    assert 0.0 <= result[0]["score"] <= 1.0


def test_rerank_empty_candidates():
    result = rerank("query", [], top_k=5)
    assert result == []


@patch("src.reranker._load_cross_encoder")
def test_rerank_passes_correct_pairs(mock_load):
    model = MagicMock()
    model.predict.return_value = [0.5, 0.6]
    mock_load.return_value = model

    candidates = [
        {"id": "d1", "text": "first doc", "score": 0.1},
        {"id": "d2", "text": "second doc", "score": 0.2},
    ]
    rerank("my question", candidates, top_k=2)

    model.predict.assert_called_once_with([
        ["my question", "first doc"],
        ["my question", "second doc"],
    ])
