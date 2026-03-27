#!/usr/bin/env python3
"""CLI for running retrieval evaluation — vector search strategies.

Usage:
    python scripts/eval_retrieval.py                  # all strategies table
    python scripts/eval_retrieval.py --top-k 10       # custom top_k
    python scripts/eval_retrieval.py --dataset path   # custom dataset
    python scripts/eval_retrieval.py --ab A B         # A/B between two named strategies
    python scripts/eval_retrieval.py --json            # machine-readable output
    python scripts/eval_retrieval.py --reranker        # include cross-encoder (needs model)
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from src.evaluation import (
    _compute_all_metrics,
    _avg_metrics,
    _pseudo_vector_results,
    evaluate_ab,
    load_gold_dataset,
)


# ── Strategy definitions ─────────────────────────────────────────────────────
# Each: (question, corpus, ids, top_k) -> list[str]


def strategy_vector(question, corpus, ids, top_k):
    """Vector search (pseudo-vector via token overlap)."""
    hits = _pseudo_vector_results(question, top_k=top_k, corpus=corpus)
    return [h["id"] for h in hits]


def strategy_compression(question, corpus, ids, top_k):
    """Vector search + extractive compression."""
    from src.compressor import compress_chunks

    hits = _pseudo_vector_results(question, top_k=top_k, corpus=corpus)
    compressed = compress_chunks(question, hits, method="extractive", max_sentences=3)
    return [h["id"] for h in compressed]


def _make_reranker_strategy():
    """Create cross-encoder reranker strategy (lazy load)."""
    def strategy_reranker(question, corpus, ids, top_k):
        """Vector search + cross-encoder reranking."""
        from src.reranker import rerank as ce_rerank

        hits = _pseudo_vector_results(question, top_k=top_k * 3, corpus=corpus)
        for item in hits:
            if "text" not in item:
                id_to_text = {c["id"]: c["text"] for c in corpus}
                item["text"] = id_to_text.get(item["id"], "")
        reranked = ce_rerank(question, hits, top_k=top_k)
        return [h["id"] for h in reranked]

    return strategy_reranker


def _make_combined_strategy():
    """Create reranker + compression strategy (lazy load)."""
    def strategy_combined(question, corpus, ids, top_k):
        """Vector search + cross-encoder reranking + compression."""
        from src.reranker import rerank as ce_rerank
        from src.compressor import compress_chunks

        hits = _pseudo_vector_results(question, top_k=top_k * 3, corpus=corpus)
        for item in hits:
            if "text" not in item:
                id_to_text = {c["id"]: c["text"] for c in corpus}
                item["text"] = id_to_text.get(item["id"], "")
        reranked = ce_rerank(question, hits, top_k=top_k)
        compressed = compress_chunks(question, reranked, method="extractive", max_sentences=3)
        return [h["id"] for h in compressed]

    return strategy_combined


STRATEGIES: dict[str, tuple[str, callable]] = {
    "vector": ("Vector search", strategy_vector),
}


# ── Output helpers ───────────────────────────────────────────────────────────

METRIC_LABELS = {
    "precision_at_k": "Prec@K",
    "recall_at_k": "Rec@K",
    "ndcg_at_k": "NDCG@K",
    "mrr": "MRR",
    "avg_precision": "MAP",
}


def _print_comparison_table(results: dict[str, tuple[dict, float]], top_k: int, n_queries: int) -> None:
    print(f"\n{'=' * 72}")
    print(f"  Retrieval Strategies -- {n_queries} queries, top_k={top_k}")
    print(f"{'=' * 72}")

    metric_keys = list(METRIC_LABELS.keys())
    header = f"  {'Strategy':<28s}"
    for mk in metric_keys:
        header += f" {METRIC_LABELS[mk]:>7s}"
    header += "  Time"
    print(header)
    print(f"  {'-' * 68}")

    best = {}
    for mk in metric_keys:
        vals = [(name, metrics[mk]) for name, (metrics, _) in results.items()]
        best[mk] = max(vals, key=lambda x: x[1])[0]

    for name, (metrics, elapsed) in results.items():
        label = STRATEGIES.get(name, (name, None))[0] if name in STRATEGIES else name
        row = f"  {label:<28s}"
        for mk in metric_keys:
            val = metrics[mk]
            marker = "*" if best[mk] == name else " "
            row += f" {val:>6.4f}{marker}"
        row += f" {elapsed:>5.1f}s"
        print(row)

    if len(results) > 1:
        print(f"\n  * = best in column")
    print()


def _print_ab_result(result: dict, name_a: str, name_b: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  A/B: {name_a} vs {name_b}")
    print(f"  {result['queries']} queries, top_k={result['top_k']}")
    print(f"{'=' * 60}")

    for label, key in [("A: " + name_a, "strategy_a"), ("B: " + name_b, "strategy_b")]:
        print(f"\n  {label}")
        print(f"  {'-' * 45}")
        for mk, ml in METRIC_LABELS.items():
            print(f"  {ml:<20s} {result[key][mk]:>8.4f}")

    print(f"\n  Deltas (B - A)")
    print(f"  {'-' * 45}")
    for mk, ml in METRIC_LABELS.items():
        val = result["deltas_b_minus_a"][mk]
        sign = "+" if val > 0 else ""
        print(f"  {ml:<20s} {sign}{val:>7.4f}")

    print(f"\n  Per-query wins")
    print(f"  {'-' * 45}")
    for mk, ml in METRIC_LABELS.items():
        w = result["per_query_wins"][mk]
        print(f"  {ml:<20s} A={w['a_wins']}  B={w['b_wins']}  tie={w['ties']}")
    print()


# ── Per-query breakdown by query_type ────────────────────────────────────────

def _run_per_query_type(strategy_fn, corpus, queries, ids, top_k):
    by_type: dict[str, list[dict]] = {}
    for query in queries:
        qtype = query.get("query_type", "unknown")
        retrieved = strategy_fn(query["question"], corpus, ids, top_k)
        m = _compute_all_metrics(retrieved, query, top_k)
        by_type.setdefault(qtype, []).append(m)
    return {qt: _avg_metrics(ms) for qt, ms in by_type.items()}


def _print_per_type(by_type: dict[str, dict], strategy_name: str) -> None:
    print(f"\n  {strategy_name} -- by query_type")
    print(f"  {'-' * 55}")
    header = f"  {'Type':<12s}"
    for ml in METRIC_LABELS.values():
        header += f" {ml:>7s}"
    print(header)
    for qt in sorted(by_type):
        row = f"  {qt:<12s}"
        for mk in METRIC_LABELS:
            row += f" {by_type[qt].get(mk, 0):>7.4f}"
        print(row)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation CLI")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K cutoff (default: 5)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to gold dataset JSON")
    parser.add_argument("--ab", nargs=2, metavar=("A", "B"), help="A/B compare two strategies by name")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")
    parser.add_argument("--reranker", action="store_true", help="Include cross-encoder reranker")
    parser.add_argument("--compression", action="store_true", help="Include extractive compression")
    parser.add_argument("--all", action="store_true", dest="all_strategies", help="Include all strategies")
    parser.add_argument("--breakdown", action="store_true", help="Show per query_type breakdown")
    args = parser.parse_args()

    if args.compression or args.all_strategies:
        STRATEGIES["compression"] = ("Vector + compression", strategy_compression)

    if args.reranker or args.all_strategies:
        try:
            STRATEGIES["reranker"] = ("Vector + cross-encoder", _make_reranker_strategy())
        except Exception as e:
            print(f"  [WARN] Cross-encoder not available: {e}", file=sys.stderr)

    if args.all_strategies:
        try:
            STRATEGIES["combined"] = ("Vector + reranker + compr.", _make_combined_strategy())
        except Exception as e:
            print(f"  [WARN] Combined strategy not available: {e}", file=sys.stderr)

    corpus, queries = load_gold_dataset(args.dataset)
    ids = [item["id"] for item in corpus]

    if args.ab:
        name_a, name_b = args.ab
        if name_a not in STRATEGIES or name_b not in STRATEGIES:
            avail = ", ".join(STRATEGIES.keys())
            print(f"  Unknown strategy. Available: {avail}", file=sys.stderr)
            sys.exit(1)
        result = evaluate_ab(
            STRATEGIES[name_a][1], STRATEGIES[name_b][1],
            top_k=args.top_k, dataset_path=args.dataset,
        )
        if args.json_output:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _print_ab_result(result, name_a, name_b)
        return

    all_results: dict[str, tuple[dict, float]] = {}
    for name, (label, fn) in STRATEGIES.items():
        t0 = time.time()
        per_query = []
        for query in queries:
            retrieved = fn(query["question"], corpus, ids, args.top_k)
            per_query.append(_compute_all_metrics(retrieved, query, args.top_k))
        avg = _avg_metrics(per_query)
        elapsed = time.time() - t0
        all_results[name] = (avg, elapsed)

    if args.json_output:
        out = {name: metrics for name, (metrics, _) in all_results.items()}
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        _print_comparison_table(all_results, args.top_k, len(queries))

        if args.breakdown:
            for name, (label, fn) in STRATEGIES.items():
                by_type = _run_per_query_type(fn, corpus, queries, ids, args.top_k)
                _print_per_type(by_type, label)
            print()


if __name__ == "__main__":
    main()
