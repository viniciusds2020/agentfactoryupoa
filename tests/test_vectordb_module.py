from pathlib import Path

from src import vectordb


def _with_temp_store(monkeypatch, tmp_path: Path, index_type: str = "flat"):
    monkeypatch.setattr(vectordb, "_stores", {})
    monkeypatch.setattr(
        "src.vectordb.get_settings",
        lambda: type(
            "S",
            (),
            {
                "chroma_path": str(tmp_path / "faiss-store"),
                "faiss_index_type": index_type,
                "faiss_hnsw_m": 32,
                "faiss_hnsw_ef_construction": 200,
                "faiss_hnsw_ef_search": 64,
            },
        )(),
    )


def test_collection_key_builds_slug():
    key = vectordb.collection_key("geral", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    assert key.startswith("default__geral__")
    assert "/" not in key


def test_resolve_query_collection_prefers_physical(monkeypatch):
    monkeypatch.setattr("src.vectordb.collection_exists", lambda name: name == "default__geral__model-x")
    resolved = vectordb.resolve_query_collection("geral", "model-x")
    assert resolved == "default__geral__model-x"


def test_resolve_query_collection_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr("src.vectordb.collection_exists", lambda name: name == "geral__model-x")
    resolved = vectordb.resolve_query_collection("geral", "model-x")
    assert resolved == "geral__model-x"


def test_upsert_and_query_roundtrip(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)

    vectordb.upsert(
        "geral",
        ids=["c1", "c2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["texto 1", "texto 2"],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d2"}],
    )

    result = vectordb.query("geral", [[1.0, 0.0]], n_results=2)
    assert result[0]["id"] == "c1"
    assert result[0]["text"] == "texto 1"
    assert "distance" in result[0]


def test_upsert_and_query_roundtrip_hnsw(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path, index_type="hnsw")

    vectordb.upsert(
        "geral",
        ids=["c1", "c2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["texto 1", "texto 2"],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d2"}],
    )

    result = vectordb.query("geral", [[1.0, 0.0]], n_results=2)
    assert result[0]["id"] == "c1"


def test_query_respects_where_filter(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "geral",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
        documents=["x", "y"],
        metadatas=[{"doc_id": "d1", "chunk_type": "child"}, {"doc_id": "d2", "chunk_type": "parent"}],
    )
    result = vectordb.query("geral", [[1.0, 0.0]], n_results=5, where={"doc_id": {"$eq": "d2"}})
    assert [r["id"] for r in result] == ["b"]


def test_get_by_metadata_supports_and_eq(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "juridico",
        ids=["x1", "x2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["t1", "t2"],
        metadatas=[
            {"doc_id": "doc-1", "parent_key": "p1", "chunk_type": "parent"},
            {"doc_id": "doc-1", "parent_key": "p1", "chunk_type": "child"},
        ],
    )
    items = vectordb.get_by_metadata(
        "juridico",
        {"$and": [{"parent_key": {"$eq": "p1"}}, {"chunk_type": {"$eq": "parent"}}]},
    )
    assert len(items) == 1
    assert items[0]["id"] == "x1"


def test_list_documents_returns_ids_and_documents(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "geral",
        ids=["a", "b"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["doc a", "doc b"],
        metadatas=[{"doc_id": "d1"}, {"doc_id": "d2"}],
    )
    ids, docs = vectordb.list_documents("geral")
    assert ids == ["a", "b"]
    assert docs == ["doc a", "doc b"]


def test_delete_by_doc_id_removes_matching_chunks(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "geral",
        ids=["c1", "c2", "c3"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        documents=["t1", "t2", "t3"],
        metadatas=[{"doc_id": "doc-a"}, {"doc_id": "doc-a"}, {"doc_id": "doc-b"}],
    )
    count = vectordb.delete_by_doc_id("geral", "doc-a")
    assert count == 2
    ids, _ = vectordb.list_documents("geral")
    assert ids == ["c3"]


def test_persistence_reload(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "geral",
        ids=["z1"],
        embeddings=[[1.0, 0.0]],
        documents=["persistido"],
        metadatas=[{"doc_id": "doc-z"}],
    )
    monkeypatch.setattr(vectordb, "_stores", {})
    result = vectordb.query("geral", [[1.0, 0.0]], n_results=1)
    assert result[0]["id"] == "z1"


def test_list_logical_collections_extracts_unique_names(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "default__juridico__model-a",
        ids=["1"],
        embeddings=[[1.0, 0.0]],
        documents=["x"],
        metadatas=[{"doc_id": "d1"}],
    )
    vectordb.upsert(
        "default__rh__model-a",
        ids=["2"],
        embeddings=[[0.0, 1.0]],
        documents=["y"],
        metadatas=[{"doc_id": "d2"}],
    )
    logical = vectordb.list_logical_collections()
    assert logical == ["juridico", "rh"]


def test_delete_collection_removes_data(monkeypatch, tmp_path: Path):
    _with_temp_store(monkeypatch, tmp_path)
    vectordb.upsert(
        "geral",
        ids=["1"],
        embeddings=[[1.0, 0.0]],
        documents=["x"],
        metadatas=[{"doc_id": "d1"}],
    )
    vectordb.delete_collection("geral")
    assert vectordb.query("geral", [[1.0, 0.0]], n_results=1) == []
