"""FAISS persistent vector store."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np

from src.config import get_settings
from src.utils import get_logger

logger = get_logger(__name__)

_stores: dict[str, "_FaissCollection"] = {}
_stores_lock = RLock()


def _storage_root() -> Path:
    settings = get_settings()
    root = Path(settings.chroma_path)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _collection_dir(name: str) -> Path:
    return _storage_root() / name


def _sanitize_filters(where: dict | None) -> dict | None:
    if not where:
        return None
    return where


def _match_eq(value: Any, expected: Any) -> bool:
    return value == expected


def _match_filter(metadata: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        terms = where.get("$and", [])
        if not isinstance(terms, list):
            return False
        return all(_match_filter(metadata, term) for term in terms)

    for key, condition in where.items():
        if isinstance(condition, dict):
            if "$eq" in condition:
                if not _match_eq(metadata.get(key), condition["$eq"]):
                    return False
            else:
                return False
        else:
            if not _match_eq(metadata.get(key), condition):
                return False
    return True


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    emb = embeddings.astype(np.float32, copy=True)
    faiss.normalize_L2(emb)
    return emb


def _build_index(vectors: np.ndarray):
    settings = get_settings()
    dim = int(vectors.shape[1])
    index_type = (settings.faiss_index_type or "flat").strip().lower()

    if index_type == "hnsw":
        m = max(4, int(settings.faiss_hnsw_m))
        ef_construction = max(16, int(settings.faiss_hnsw_ef_construction))
        ef_search = max(16, int(settings.faiss_hnsw_ef_search))
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        return index, "hnsw"

    # default
    index = faiss.IndexFlatIP(dim)
    return index, "flat"


@dataclass
class _FaissCollection:
    name: str
    ids: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    metadatas: list[dict] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    _index: faiss.IndexFlatIP | None = None
    _id_to_pos: dict[str, int] = field(default_factory=dict)

    @property
    def base_dir(self) -> Path:
        return _collection_dir(self.name)

    @property
    def data_file(self) -> Path:
        return self.base_dir / "data.json"

    @property
    def vectors_file(self) -> Path:
        return self.base_dir / "vectors.npy"

    def load(self) -> None:
        if not self.data_file.exists() or not self.vectors_file.exists():
            self._rebuild_state()
            return

        payload = json.loads(self.data_file.read_text(encoding="utf-8"))
        self.ids = list(payload.get("ids", []))
        self.documents = list(payload.get("documents", []))
        self.metadatas = list(payload.get("metadatas", []))

        vectors = np.load(self.vectors_file, allow_pickle=False)
        if vectors.size == 0:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            self.embeddings = vectors.astype(np.float32, copy=False)
        self._rebuild_state()

    def save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self.ids,
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        self.data_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        np.save(self.vectors_file, self.embeddings.astype(np.float32, copy=False))

    def _rebuild_state(self) -> None:
        self._id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
        if self.embeddings.size == 0 or len(self.ids) == 0:
            self._index = None
            return
        dim = int(self.embeddings.shape[1])
        if dim <= 0:
            self._index = None
            return
        normalized = _normalize_embeddings(self.embeddings)
        index, index_label = _build_index(normalized)
        index.add(normalized)
        self._index = index
        logger.info(f"FAISS collection '{self.name}' rebuilt with index={index_label}, vectors={len(self.ids)}")

    def upsert(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids, embeddings, documents and metadatas must have same length")

        incoming = np.asarray(embeddings, dtype=np.float32)
        if incoming.ndim != 2:
            raise ValueError("embeddings must be a 2D list")

        if self.embeddings.size == 0:
            self.embeddings = np.empty((0, incoming.shape[1]), dtype=np.float32)
        elif int(self.embeddings.shape[1]) != int(incoming.shape[1]):
            raise ValueError("embedding dimension mismatch for collection")

        for i, cid in enumerate(ids):
            if cid in self._id_to_pos:
                pos = self._id_to_pos[cid]
                self.embeddings[pos] = incoming[i]
                self.documents[pos] = documents[i]
                self.metadatas[pos] = metadatas[i] or {}
            else:
                self.ids.append(cid)
                self.documents.append(documents[i])
                self.metadatas.append(metadatas[i] or {})
                self.embeddings = np.vstack([self.embeddings, incoming[i : i + 1]])

        self._rebuild_state()
        self.save()

    def query(self, query_embeddings: list[list[float]], n_results: int, where: dict | None) -> list[dict]:
        if not query_embeddings or len(self.ids) == 0:
            return []

        q = np.asarray(query_embeddings, dtype=np.float32)
        if q.ndim != 2 or q.shape[0] == 0:
            return []

        filtered_positions = [i for i, meta in enumerate(self.metadatas) if _match_filter(meta or {}, where)]
        if not filtered_positions:
            return []

        q_norm = _normalize_embeddings(q[:1].copy())
        top_k = min(max(n_results, 1), len(filtered_positions))

        if where is None and self._index is not None:
            sims, pos = self._index.search(q_norm, top_k)
            positions = [int(p) for p in pos[0] if p >= 0]
            distances = [float(1.0 - s) for s in sims[0][: len(positions)]]
        else:
            vectors = self.embeddings[filtered_positions]
            tmp_index = faiss.IndexFlatIP(int(vectors.shape[1]))
            tmp_index.add(_normalize_embeddings(vectors.copy()))
            sims, rel_pos = tmp_index.search(q_norm, top_k)
            rel = [int(p) for p in rel_pos[0] if p >= 0]
            positions = [filtered_positions[p] for p in rel]
            distances = [float(1.0 - s) for s in sims[0][: len(positions)]]

        chunks: list[dict] = []
        for p, dist in zip(positions, distances):
            chunks.append(
                {
                    "id": self.ids[p],
                    "text": self.documents[p],
                    "metadata": self.metadatas[p] or {},
                    "distance": dist,
                }
            )
        return chunks

    def get(self, where: dict | None, include: list[str]) -> dict:
        positions = [i for i, meta in enumerate(self.metadatas) if _match_filter(meta or {}, where)]
        result: dict[str, Any] = {"ids": [self.ids[i] for i in positions]}
        if "documents" in include:
            result["documents"] = [self.documents[i] for i in positions]
        if "metadatas" in include:
            result["metadatas"] = [self.metadatas[i] for i in positions]
        return result

    def delete_by_where(self, where: dict) -> int:
        to_keep = [i for i, meta in enumerate(self.metadatas) if not _match_filter(meta or {}, where)]
        deleted = len(self.ids) - len(to_keep)
        if deleted <= 0:
            return 0

        self.ids = [self.ids[i] for i in to_keep]
        self.documents = [self.documents[i] for i in to_keep]
        self.metadatas = [self.metadatas[i] for i in to_keep]
        if len(to_keep) == 0:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
        else:
            self.embeddings = self.embeddings[to_keep]
        self._rebuild_state()
        self.save()
        return deleted

    def clear(self) -> None:
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self._rebuild_state()
        for path in (self.data_file, self.vectors_file):
            if path.exists():
                path.unlink()
        if self.base_dir.exists():
            try:
                self.base_dir.rmdir()
            except OSError:
                pass


def _get_collection(name: str) -> _FaissCollection | None:
    with _stores_lock:
        if name in _stores:
            return _stores[name]
        col_dir = _collection_dir(name)
        if not (col_dir / "data.json").exists():
            return None
        col = _FaissCollection(name=name)
        col.load()
        _stores[name] = col
        return col


def _get_or_create_collection(name: str) -> _FaissCollection:
    with _stores_lock:
        if name in _stores:
            return _stores[name]
        col = _FaissCollection(name=name)
        col.load()
        _stores[name] = col
        return col


def collection_key(collection_name: str, embedding_model: str, workspace_id: str = "default") -> str:
    """Build a stable physical collection name per logical collection and embedding model."""
    workspace_slug = re.sub(r"[^a-z0-9]+", "-", workspace_id.lower()).strip("-")
    model_slug = re.sub(r"[^a-z0-9]+", "-", embedding_model.lower()).strip("-")
    return f"{workspace_slug}__{collection_name}__{model_slug}"


def collection_exists(name: str) -> bool:
    col = _get_collection(name)
    return col is not None and len(col.ids) > 0


def resolve_query_collection(collection_name: str, embedding_model: str, workspace_id: str = "default") -> str:
    """Resolve the best collection to query, preserving compatibility with legacy data."""
    physical_name = collection_key(collection_name, embedding_model, workspace_id=workspace_id)
    if collection_exists(physical_name):
        return physical_name
    legacy_physical_name = f"{collection_name}__{re.sub(r'[^a-z0-9]+', '-', embedding_model.lower()).strip('-')}"
    if collection_exists(legacy_physical_name):
        return legacy_physical_name
    if workspace_id == "default" and collection_exists(collection_name):
        return collection_name
    return physical_name


def get_or_create_collection(name: str) -> _FaissCollection:
    return _get_or_create_collection(name)


def upsert(
    collection_name: str,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    col = get_or_create_collection(collection_name)
    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    logger.info(f"Upserted {len(ids)} chunks into '{collection_name}'")


def query(
    collection_name: str,
    query_embeddings: list[list[float]],
    n_results: int = 10,
    where: dict | None = None,
) -> list[dict]:
    col = _get_collection(collection_name)
    if col is None:
        return []
    return col.query(query_embeddings=query_embeddings, n_results=n_results, where=_sanitize_filters(where))


def list_documents(collection_name: str) -> tuple[list[str], list[str]]:
    """Return all chunk ids and documents from a collection."""
    col = _get_collection(collection_name)
    if col is None:
        return [], []
    result = col.get(where=None, include=["documents"])
    return result["ids"], result.get("documents", [])


def delete_by_doc_id(collection_name: str, doc_id: str) -> int:
    """Delete all chunks belonging to a document. Returns count deleted."""
    col = _get_collection(collection_name)
    if col is None:
        return 0
    count = col.delete_by_where({"doc_id": {"$eq": doc_id}})
    if count > 0:
        logger.info(f"Deleted {count} old chunks for doc_id='{doc_id}' from '{collection_name}'")
    return count


def get_by_metadata(
    collection_name: str,
    where: dict,
    include: list[str] | None = None,
) -> list[dict]:
    """Fetch chunks matching a metadata filter. Used for parent-child retrieval."""
    col = _get_collection(collection_name)
    if col is None:
        return []
    inc = include or ["documents", "metadatas"]
    results = col.get(where=where, include=inc)
    items: list[dict] = []
    for i, cid in enumerate(results["ids"]):
        item: dict[str, Any] = {"id": cid}
        if "documents" in inc and results.get("documents"):
            item["text"] = results["documents"][i]
        if "metadatas" in inc and results.get("metadatas"):
            item["metadata"] = results["metadatas"][i]
        items.append(item)
    return items


def list_logical_collections() -> list[str]:
    """Return sorted unique logical collection names from physical collection dirs."""
    root = _storage_root()
    logical: set[str] = set()
    for col_dir in root.iterdir():
        if not col_dir.is_dir():
            continue
        name = col_dir.name
        parts = name.split("__")
        if len(parts) >= 3:
            logical.add(parts[1])
        elif "__" in name:
            logical.add(name.split("__", 1)[0])
        else:
            logical.add(name)
    return sorted(logical)


def delete_collection(name: str) -> None:
    col = _get_collection(name)
    if col is None:
        col_dir = _collection_dir(name)
        if col_dir.exists():
            for file in col_dir.glob("*"):
                file.unlink()
            col_dir.rmdir()
        return
    col.clear()
    with _stores_lock:
        _stores.pop(name, None)
    logger.info(f"Deleted collection '{name}'")
