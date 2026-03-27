#!/usr/bin/env python3
"""Batch FAISS reindex script.

Reindexes documents registered in control plane by resolving files from a docs directory.

Examples:
  python scripts/reindex_faiss.py
  python scripts/reindex_faiss.py --docs-dir data/docs --workspace-id default
  python scripts/reindex_faiss.py --collection geral --embedding-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src import ingestion, vectordb
from src.controlplane import list_documents, upsert_document
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ReindexResult:
    attempted: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0


def _build_file_maps(docs_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for p in docs_dir.rglob("*"):
        if not p.is_file():
            continue
        by_name.setdefault(p.name.lower(), []).append(p)
        by_stem.setdefault(p.stem.lower(), []).append(p)
    return by_name, by_stem


def _pick_path(
    docs_dir: Path,
    by_name: dict[str, list[Path]],
    by_stem: dict[str, list[Path]],
    filename: str,
    doc_id: str,
) -> Path | None:
    candidates_exact = [
        filename,
        Path(filename).name if filename else "",
        doc_id,
        Path(doc_id).name if doc_id else "",
    ]
    for key in candidates_exact:
        key = (key or "").strip().lower()
        if not key:
            continue
        hits = by_name.get(key, [])
        if hits:
            return hits[0]
        direct = docs_dir / key
        if direct.exists() and direct.is_file():
            return direct

    candidates_stem = [
        Path(filename).stem if filename else "",
        Path(doc_id).stem if doc_id else "",
    ]
    for stem in candidates_stem:
        stem = (stem or "").strip().lower()
        if not stem:
            continue
        hits = by_stem.get(stem, [])
        if hits:
            return hits[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch reindex documents into FAISS.")
    parser.add_argument("--workspace-id", default="default", help="Workspace id to reindex.")
    parser.add_argument("--docs-dir", default="data/docs", help="Directory with source files.")
    parser.add_argument("--collection", default="", help="Optional filter by collection.")
    parser.add_argument("--embedding-model", default="", help="Optional override embedding model.")
    parser.add_argument("--domain-profile", default="", help="Optional domain profile override.")
    parser.add_argument(
        "--clear-collections",
        action="store_true",
        help="Delete target physical collections before reindexing.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists() or not docs_dir.is_dir():
        raise SystemExit(f"docs directory not found: {docs_dir}")

    records = list_documents(args.workspace_id, collection=args.collection or None)
    if not records:
        print("No document records found for selected filters.")
        return

    by_name, by_stem = _build_file_maps(docs_dir)
    result = ReindexResult()
    cleared: set[str] = set()

    print(f"Reindex start: workspace={args.workspace_id}, records={len(records)}, docs_dir={docs_dir}")
    for rec in records:
        result.attempted += 1
        model_name = args.embedding_model or rec.embedding_model
        physical_collection = vectordb.collection_key(rec.collection, model_name, workspace_id=rec.workspace_id)

        if args.clear_collections and physical_collection not in cleared:
            vectordb.delete_collection(physical_collection)
            cleared.add(physical_collection)

        source_path = _pick_path(docs_dir, by_name, by_stem, rec.filename, rec.doc_id)
        if not source_path:
            result.skipped += 1
            upsert_document(
                workspace_id=rec.workspace_id,
                collection=rec.collection,
                doc_id=rec.doc_id,
                filename=rec.filename,
                embedding_model=model_name,
                status="failed",
                chunks_indexed=0,
                error="source file not found for reindex",
            )
            print(f"[SKIP] {rec.collection}/{rec.doc_id}: source not found")
            continue

        try:
            chunks = ingestion.ingest(
                collection=rec.collection,
                source=str(source_path),
                doc_id=rec.doc_id,
                embedding_model=model_name,
                workspace_id=rec.workspace_id,
                domain_profile=args.domain_profile or None,
            )
            upsert_document(
                workspace_id=rec.workspace_id,
                collection=rec.collection,
                doc_id=rec.doc_id,
                filename=rec.filename,
                embedding_model=model_name,
                status="indexed",
                chunks_indexed=chunks,
                error="",
            )
            result.indexed += 1
            print(f"[OK]   {rec.collection}/{rec.doc_id}: {chunks} chunks")
        except Exception as exc:
            result.failed += 1
            upsert_document(
                workspace_id=rec.workspace_id,
                collection=rec.collection,
                doc_id=rec.doc_id,
                filename=rec.filename,
                embedding_model=model_name,
                status="failed",
                chunks_indexed=0,
                error=str(exc),
            )
            logger.exception("Reindex failed", extra={"props": {"doc_id": rec.doc_id, "collection": rec.collection}})
            print(f"[FAIL] {rec.collection}/{rec.doc_id}: {exc}")

    print("\nReindex summary:")
    print(f"  attempted: {result.attempted}")
    print(f"  indexed:   {result.indexed}")
    print(f"  skipped:   {result.skipped}")
    print(f"  failed:    {result.failed}")


if __name__ == "__main__":
    main()
