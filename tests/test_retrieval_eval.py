"""Retrieval evaluation tests — gold dataset, vector search quality."""
from src.evaluation import _pseudo_vector_results, precision_at_k, recall_at_k, GOLD_CORPUS, GOLD_QUERIES


def test_precision_at_k_calculation():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], k=3) == 2 / 3
    assert precision_at_k(["x", "y"], ["a"], k=2) == 0.0
    assert precision_at_k(["a"], ["a"], k=1) == 1.0


def test_recall_at_k_calculation():
    assert recall_at_k(["a", "b"], ["a", "c"], k=2) == 0.5
    assert recall_at_k(["a", "c"], ["a", "c"], k=2) == 1.0
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_pseudo_vector_recall_at_5_on_gold_dataset():
    """Pseudo-vector should find the relevant chunk in top-5 for each gold query."""
    total_recall = 0.0
    for query in GOLD_QUERIES:
        results = _pseudo_vector_results(query["question"], top_k=5)
        retrieved_ids = [r["id"] for r in results]
        total_recall += recall_at_k(retrieved_ids, query["relevant"], k=5)

    avg_recall = total_recall / len(GOLD_QUERIES)
    assert avg_recall >= 0.6, f"Vector avg recall@5 = {avg_recall:.2f}, expected >= 0.60"


def test_format_context_citation_mapping():
    """Verify [N] citations map correctly to source documents."""
    from src.prompts import format_context

    items = [
        {"id": "c1", "text": "Trecho A.", "metadata": {"doc_id": "doc-alpha.pdf", "page_number": 5}},
        {"id": "c2", "text": "Trecho B.", "metadata": {"doc_id": "doc-beta.pdf"}},
        {"id": "c3", "text": "Trecho C.", "metadata": {"doc_id": "doc-gamma.pdf", "page_number": 12}},
    ]
    ctx = format_context(items)

    assert "[1]" in ctx and "doc-alpha.pdf" in ctx
    assert "[2]" in ctx and "doc-beta.pdf" in ctx
    assert "[3]" in ctx and "doc-gamma.pdf" in ctx
    assert "p. 5" in ctx
    assert "p. 12" in ctx
    line_2 = [l for l in ctx.split("\n") if "[2]" in l][0]
    assert "p." not in line_2
