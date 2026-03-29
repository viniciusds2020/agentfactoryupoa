from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient

import app as app_module


client = TestClient(app_module.app)


def test_ingest_rejects_large_file(monkeypatch):
    # Temporarily lower the limit so we don't need to allocate 2 GB+
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 1024)
    too_large = BytesIO(b"x" * 1025)
    response = client.post(
        "/ingest",
        files={"file": ("big.txt", too_large, "text/plain")},
        data={"collection": "geral", "embedding_model": "intfloat/multilingual-e5-small"},
    )
    assert response.status_code == 413
    assert "2 GB" in response.text


def test_ingest_passes_embedding_model(monkeypatch):
    captured = {}

    def fake_ingest(collection, source, doc_id=None, embedding_model=None, workspace_id="default", domain_profile=None):
        captured["collection"] = collection
        captured["doc_id"] = doc_id
        captured["embedding_model"] = embedding_model
        captured["workspace_id"] = workspace_id
        captured["domain_profile"] = domain_profile
        return 3

    monkeypatch.setattr("src.ingestion.ingest", fake_ingest)

    response = client.post(
        "/ingest",
        files={"file": ("ok.txt", BytesIO(b"conteudo"), "text/plain")},
        data={"collection": "geral", "embedding_model": "BAAI/bge-m3"},
    )

    assert response.status_code == 200
    assert captured["collection"] == "geral"
    assert captured["embedding_model"] == "BAAI/bge-m3"
    assert captured["doc_id"] == "ok.txt"
    assert captured["workspace_id"] == "default"


def test_patch_collection_context_updates_base(monkeypatch):
    monkeypatch.setattr("app.update_collection_context", lambda workspace_id, collection, context_hint: 1)

    response = client.patch(
        "/collections/base_csv/context",
        json={"context_hint": "Base de clientes com renda mensal e localizacao"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection"] == "base_csv"
    assert payload["updated_documents"] == 1
    assert "clientes" in payload["context_hint"]


def test_get_collection_semantic_profile(monkeypatch):
    monkeypatch.setattr(
        "app.get_table_profile",
        lambda workspace_id, collection: {
            "workspace_id": workspace_id,
            "collection": collection,
            "table_name": "cadastro_records",
            "base_context": "Base de clientes",
            "subject_label": "clientes",
            "created_at": "2026-03-29 10:00:00",
            "updated_at": "2026-03-29 10:00:00",
        },
    )
    monkeypatch.setattr(
        "app.list_column_profiles",
        lambda workspace_id, collection: [
            {
                "workspace_id": workspace_id,
                "collection": collection,
                "column_name": "renda_mensal",
                "display_name": "renda_mensal",
                "physical_type": "numeric",
                "semantic_type": "measure_currency",
                "role": "metric",
                "unit": "brl",
                "aliases": ["renda"],
                "examples": ["1000"],
                "description": "Medida monetaria.",
                "cardinality": 100,
                "allowed_operations": ["sum", "avg"],
            }
        ],
    )

    response = client.get("/collections/base_csv/semantic-profile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection"] == "base_csv"
    assert payload["profile"]["subject_label"] == "clientes"
    assert payload["columns"][0]["semantic_type"] == "measure_currency"


def test_ingest_returns_422_with_underlying_error_detail(monkeypatch):
    monkeypatch.setattr("src.ingestion.ingest", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("parser failure")))

    response = client.post(
        "/ingest",
        files={"file": ("ok.txt", BytesIO(b"conteudo"), "text/plain")},
        data={"collection": "geral", "embedding_model": "BAAI/bge-m3"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "parser failure"


def test_ingest_cleanup_error_does_not_mask_result(monkeypatch):
    monkeypatch.setattr("src.ingestion.ingest", lambda *args, **kwargs: 1)
    monkeypatch.setattr("os.unlink", lambda path: (_ for _ in ()).throw(PermissionError("locked")))

    response = client.post(
        "/ingest",
        files={"file": ("ok.txt", BytesIO(b"conteudo"), "text/plain")},
        data={"collection": "geral", "embedding_model": "BAAI/bge-m3"},
    )

    assert response.status_code == 200
    assert response.json()["chunks_indexed"] == 1


def test_unhandled_ingest_error_returns_exact_detail_in_development(monkeypatch):
    client_no_raise = TestClient(app_module.app, raise_server_exceptions=False)
    monkeypatch.setattr("app._validate_upload", lambda file: (_ for _ in ()).throw(RuntimeError("docx parser crashed")))

    response = client_no_raise.post(
        "/ingest",
        files={"file": ("ok.docx", BytesIO(b"conteudo"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"collection": "geral", "embedding_model": "BAAI/bge-m3"},
    )

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "RuntimeError: docx parser crashed"
    assert "request_id" in payload


def test_chat_message_uses_requested_embedding_for_new_conversation(monkeypatch):
    monkeypatch.setattr("src.history.create_conversation", lambda workspace_id, collection, embedding_model, title="": "conv-1")

    class Conv:
        id = "conv-1"
        workspace_id = "default"
        title = "Conversa 19/03 10:00"
        collection = "geral"
        embedding_model = "BAAI/bge-m3"
        created_at = ""
        updated_at = ""

    monkeypatch.setattr("src.history.get_conversation", lambda conv_id, workspace_id=None: Conv())
    monkeypatch.setattr("src.history.rename_conversation", lambda conv_id, title: None)
    monkeypatch.setattr("src.history.save_message", lambda *args, **kwargs: None)

    captured = {}

    class Result:
        answer = "ok"
        request_id = "req-1"
        sources = []

    def fake_answer(collection, question, history=None, request_id="-", where=None, embedding_model=None, workspace_id="default", domain_profile=None):
        captured["collection"] = collection
        captured["question"] = question
        captured["embedding_model"] = embedding_model
        captured["workspace_id"] = workspace_id
        captured["domain_profile"] = domain_profile
        return Result()

    monkeypatch.setattr(app_module, "answer", fake_answer)

    response = client.post(
        "/chat/message",
        json={
            "collection": "geral",
            "question": "teste",
            "history": [],
            "embedding_model": "BAAI/bge-m3",
            "domain_profile": "legal",
        },
    )

    assert response.status_code == 200
    assert captured["embedding_model"] == "BAAI/bge-m3"
    assert captured["workspace_id"] == "default"
    assert captured["domain_profile"] == "legal"


def test_chat_returns_422_for_invalid_question(monkeypatch):
    response = client.post(
        "/chat",
        json={
            "collection": "geral",
            "question": "   ",
            "history": [],
            "embedding_model": "BAAI/bge-m3",
        },
    )

    assert response.status_code == 422
    assert "pergunta" in response.json()["detail"].lower()


def test_chat_message_rejects_invalid_question_before_persisting(monkeypatch):
    called = {"saved": False}

    monkeypatch.setattr("src.history.save_message", lambda *args, **kwargs: called.__setitem__("saved", True))
    monkeypatch.setattr("src.history.create_conversation", lambda *args, **kwargs: "conv-1")
    monkeypatch.setattr(
        "src.history.get_conversation",
        lambda conv_id, workspace_id=None: type("Conv", (), {"id": "conv-1", "workspace_id": "default", "title": "Conversa 20/03 10:00", "embedding_model": "model-a"})(),
    )

    response = client.post(
        "/chat/message",
        json={
            "collection": "geral",
            "question": "   ",
            "history": [],
            "embedding_model": "BAAI/bge-m3",
        },
    )

    assert response.status_code == 422
    assert called["saved"] is False


def test_create_conversation_validates_collection_name():
    response = client.post(
        "/conversations",
        json={"collection": "colecao invalida", "embedding_model": "BAAI/bge-m3"},
    )

    assert response.status_code == 422
    assert "cole" in response.json()["detail"].lower()


def test_get_messages_preserves_source_metadata(monkeypatch):
    class StoredMessage:
        role = "assistant"
        content = "resposta"
        sources = [
            {
                "chunk_id": "c1",
                "doc_id": "doc-1",
                "excerpt": "trecho",
                "score": 0.4,
                "metadata": {"page_number": 7, "source_filename": "contrato.pdf"},
            }
        ]

    monkeypatch.setattr(
        "src.history.get_conversation",
        lambda conv_id, workspace_id=None: type(
            "Conv",
            (),
            {"id": conv_id, "workspace_id": "default", "title": "t", "collection": "geral", "embedding_model": "m", "created_at": "", "updated_at": ""},
        )(),
    )
    monkeypatch.setattr("src.history.load_messages", lambda conv_id: [StoredMessage()])

    response = client.get("/conversations/conv-1/messages")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["sources"][0]["page_number"] == 7
    assert payload[0]["sources"][0]["source_filename"] == "contrato.pdf"


def test_workspaces_endpoint_lists_default_workspace():
    response = client.get("/workspaces")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 1
    assert payload[0]["id"]


def test_async_ingest_returns_job(monkeypatch):
    monkeypatch.setattr("app._process_ingestion_job", lambda **kwargs: None)

    response = client.post(
        "/ingest/async",
        files={"file": ("ok.txt", BytesIO(b"conteudo"), "text/plain")},
        data={"collection": "geral", "embedding_model": "BAAI/bge-m3"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["collection"] == "geral"


def test_conversations_are_scoped_by_workspace():
    workspace_response = client.post("/workspaces", json={"name": f"workspace-test-isolation-{uuid4().hex[:8]}"})
    assert workspace_response.status_code == 200
    workspace = workspace_response.json()
    headers = {"X-API-Key": workspace["api_key"]}

    unique_title = "Conv Workspace Exclusiva"
    create_response = client.post(
        "/conversations",
        json={"collection": "geral", "embedding_model": "BAAI/bge-m3", "title": unique_title},
        headers=headers,
    )
    assert create_response.status_code == 200

    scoped = client.get(f"/conversations?q={unique_title}", headers=headers)
    assert scoped.status_code == 200
    assert any(item["title"] == unique_title for item in scoped.json())

    default_scope = client.get(f"/conversations?q={unique_title}")
    assert default_scope.status_code == 200
    assert all(item["title"] != unique_title for item in default_scope.json())


def test_observability_endpoint_returns_metrics_shape():
    response = client.get("/observability")
    assert response.status_code == 200
    payload = response.json()
    assert "counters" in payload
    assert "timers_ms" in payload


def test_settings_endpoint_exposes_domain_profiles():
    response = client.get("/settings")
    assert response.status_code == 200
    payload = response.json()
    assert "default_domain_profile" in payload
    assert "available_domain_profiles" in payload
    assert "general" in payload["available_domain_profiles"]


def test_retrieval_evaluation_endpoint_returns_snapshot():
    response = client.get("/evaluation/retrieval?top_k=3")
    assert response.status_code == 200
    payload = response.json()
    assert payload["top_k"] == 3
    assert "vector" in payload


def test_tabular_evaluation_endpoint_returns_snapshot(monkeypatch):
    monkeypatch.setattr(
        "app.evaluate_tabular_benchmark",
        lambda: {
            "cases": 7,
            "summary": {"tabular_plan_success_rate": 1.0},
            "details": [{"question": "Qual e a media de idade dos clientes do Ceara?"}],
        },
    )

    response = client.get("/evaluation/tabular")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cases"] == 7
    assert payload["summary"]["tabular_plan_success_rate"] == 1.0
