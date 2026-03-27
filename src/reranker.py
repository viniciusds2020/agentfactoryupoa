"""Cross-encoder reranking for retrieval results."""
from __future__ import annotations

from functools import lru_cache
from math import exp
from typing import Any

from src.utils import get_logger, log_event

logger = get_logger(__name__)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> Any:
    """Lazy-load a cross-encoder model. Cached to avoid re-downloading."""
    from sentence_transformers import CrossEncoder

    log_event(logger, 20, "Loading cross-encoder model", model=model_name)
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 10,
) -> list[dict]:
    """Rerank candidates using a cross-encoder model.

    Each candidate must have at least a ``"text"`` key.
    Returns candidates sorted by cross-encoder score (descending), limited to ``top_k``.
    The cross-encoder score is stored in ``candidate["score"]``.
    """
    if not candidates:
        return []

    model = _load_cross_encoder(model_name)

    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        base_score = float(candidate.get("score", 0.0))
        ce_score = float(score)
        ce_norm = _sigmoid(ce_score)
        candidate["base_score"] = base_score
        candidate["ce_score"] = ce_score
        candidate["ce_score_norm"] = ce_norm
        # Keep final score in 0..1 so retrieval_min_score remains meaningful.
        candidate["score"] = (0.65 * base_score) + (0.35 * ce_norm)

    reranked = sorted(candidates, key=lambda c: c["score"], reverse=True)

    log_event(
        logger, 20, "Cross-encoder reranking done",
        model=model_name,
        input_count=len(candidates),
        output_count=min(len(reranked), top_k),
    )

    return reranked[:top_k]
