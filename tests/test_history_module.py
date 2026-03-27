from pathlib import Path

from src import history


def test_history_crud_flow(tmp_path, monkeypatch):
    db_path = tmp_path / "history.db"
    monkeypatch.setattr(history, "_DB_PATH", Path(db_path))
    monkeypatch.setattr("src.history.log_event", lambda *args, **kwargs: None)

    history.init_db()

    conv_id = history.create_conversation("default", "juridico", "model-a", title="Minha conversa")
    conv = history.get_conversation(conv_id, "default")
    assert conv is not None
    assert conv.title == "Minha conversa"
    assert conv.collection == "juridico"
    assert conv.embedding_model == "model-a"

    history.rename_conversation(conv_id, "Novo título")
    renamed = history.get_conversation(conv_id, "default")
    assert renamed is not None
    assert renamed.title == "Novo título"

    history.save_message(conv_id, "user", "pergunta", [{"chunk_id": "c1", "doc_id": "d1", "excerpt": "x", "score": 0.9}])
    history.save_message(conv_id, "assistant", "resposta", [])

    messages = history.load_messages(conv_id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].sources[0]["doc_id"] == "d1"

    listed = history.list_conversations("default")
    assert any(item.id == conv_id for item in listed)

    searched = history.search_conversations("default", "pergunta")
    assert any(item.id == conv_id for item in searched)

    history.delete_conversation(conv_id)
    assert history.get_conversation(conv_id, "default") is None


def test_create_conversation_without_title_generates_default(tmp_path, monkeypatch):
    db_path = tmp_path / "history_default.db"
    monkeypatch.setattr(history, "_DB_PATH", Path(db_path))
    monkeypatch.setattr("src.history.log_event", lambda *args, **kwargs: None)

    history.init_db()
    conv_id = history.create_conversation("default", "rh", "model-b")
    conv = history.get_conversation(conv_id, "default")
    assert conv is not None
    assert conv.title.startswith("Conversa ")


def test_src_to_dict_handles_object():
    class FakeSource:
        chunk_id = "c1"
        doc_id = "d1"
        excerpt = "trecho"
        score = 0.42
        metadata = {"a": 1}

    result = history._src_to_dict(FakeSource())
    assert result["chunk_id"] == "c1"
    assert result["metadata"] == {"a": 1}


def test_src_to_dict_returns_existing_dict_unchanged():
    payload = {"chunk_id": "c9", "metadata": {"ok": True}}
    assert history._src_to_dict(payload) is payload


def test_load_messages_and_search_return_empty_for_unknown_conversation(tmp_path, monkeypatch):
    db_path = tmp_path / "history_empty.db"
    monkeypatch.setattr(history, "_DB_PATH", Path(db_path))
    monkeypatch.setattr("src.history.log_event", lambda *args, **kwargs: None)

    history.init_db()

    assert history.get_conversation("missing", "default") is None
    assert history.load_messages("missing") == []
    assert history.search_conversations("default", "nao-existe") == []


def test_purge_old_conversations_deletes_expired(tmp_path, monkeypatch):
    db_path = tmp_path / "history_purge.db"
    monkeypatch.setattr(history, "_DB_PATH", Path(db_path))
    monkeypatch.setattr("src.history.log_event", lambda *args, **kwargs: None)

    history.init_db()

    # Create a conversation and manually backdate it
    conv_id = history.create_conversation("default", "rh", "model-a", title="Conversa antiga")
    import sqlite3
    from contextlib import closing
    conn = sqlite3.connect(str(db_path))
    with closing(conn):
        with conn:
            conn.execute(
                "UPDATE conversations SET updated_at = datetime('now', 'localtime', '-100 days') WHERE id = ?",
                (conv_id,),
            )

    # Create a recent conversation that should survive
    recent_id = history.create_conversation("default", "rh", "model-a", title="Conversa recente")

    deleted = history.purge_old_conversations(days=90)
    assert deleted == 1
    assert history.get_conversation(conv_id, "default") is None
    assert history.get_conversation(recent_id, "default") is not None
