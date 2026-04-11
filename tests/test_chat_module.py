from src import chat
from pathlib import Path
import json


def _settings(**overrides):
    values = {
        "retrieval_top_k": 3,
        "retrieval_min_score": 0.0,
        "embedding_model": "default-model",
        "default_domain_profile": "general",
        "max_context_tokens": 6000,
        "chars_per_token": 3.5,
        "query_expansion_enabled": False,
        "query_expansion_max_reformulations": 2,
        "hyde_enabled": False,
        "hyde_merge_original": True,
        "reranker_enabled": False,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "reranker_top_k": 10,
        "compression_enabled": False,
        "compression_method": "extractive",
        "compression_max_sentences": 3,
    }
    values.update(overrides)
    return type("S", (), values)()


def test_answer_uses_embedding_model_and_history(monkeypatch):
    monkeypatch.setattr("src.chat.get_settings", lambda: _settings())
    monkeypatch.setattr("src.chat.vectordb.resolve_query_collection", lambda c, m, workspace_id="default": f"{workspace_id}::{c}::{m}")
    monkeypatch.setattr("src.chat.llm.embed", lambda texts, model_name=None: [[0.1, 0.2]])
    monkeypatch.setattr(
        "src.chat.vectordb.query",
        lambda collection_name, query_embeddings, n_results=10, where=None: [
            {"id": "chunk-1", "text": "Trecho relevante", "metadata": {"doc_id": "doc-a"}, "distance": 0.1}
        ],
    )

    captured = {}

    def fake_chat(messages, system=""):
        captured["messages"] = messages
        captured["system"] = system
        return "Resposta final"

    monkeypatch.setattr("src.chat.llm.chat", fake_chat)

    history = [chat.ChatMessage(role="user", content="mensagem anterior")]
    result = chat.answer(
        collection="geral",
        question="Qual e a resposta?",
        history=history,
        embedding_model="custom-model",
        request_id="req-1",
    )

    assert result.answer == "Resposta final"
    assert result.sources[0].doc_id == "doc-a"
    assert captured["messages"][0]["content"] == "mensagem anterior"
    assert "Qual e a resposta?" in captured["messages"][-1]["content"]
    assert "dominio" in captured["system"].lower()


def test_filter_by_threshold_removes_low_scores():
    results = [
        {"id": "c1", "text": "bom", "score": 0.02},
        {"id": "c2", "text": "medio", "score": 0.01},
        {"id": "c3", "text": "ruim", "score": 0.002},
    ]
    filtered = chat._filter_by_threshold(results, min_score=0.005)
    assert len(filtered) == 2
    assert all(r["score"] >= 0.005 for r in filtered)


def test_filter_by_threshold_keeps_top1_when_all_filtered():
    results = [
        {"id": "c1", "text": "baixo", "score": 0.001},
        {"id": "c2", "text": "baixo", "score": 0.0005},
    ]
    filtered = chat._filter_by_threshold(results, min_score=0.1)
    assert len(filtered) == 1
    assert filtered[0]["id"] == "c1"


def test_filter_by_threshold_disabled_when_zero():
    results = [
        {"id": "c1", "text": "a", "score": 0.001},
        {"id": "c2", "text": "b", "score": 0.0001},
    ]
    filtered = chat._filter_by_threshold(results, min_score=0.0)
    assert len(filtered) == 2


def test_keyword_tokens_keep_structural_numerals_for_chapter_queries():
    tokens = chat._keyword_tokens("Resuma o capitulo 1")
    assert "1" in tokens
    assert "i" in tokens


def test_structural_match_strength_uses_exact_numeric_boundaries():
    matching = {
        "metadata": {
            "section": "CAPITULO I - DAS CARACTERISTICAS",
        }
    }
    non_matching = {
        "metadata": {
            "section": "CAPITULO XI - DISPOSICOES FINAIS",
        }
    }

    assert chat._structural_match_strength("Resuma o capitulo 1", matching) > 0
    assert chat._structural_match_strength("Resuma o capitulo 1", non_matching) == 0


def test_supplement_chapter_matches_adds_missing_target_chapter(monkeypatch):
    initial = [
        {
            "id": "wrong-v",
            "text": "CAPÍTULO V ...",
            "score": 0.4,
            "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO V"},
        }
    ]

    monkeypatch.setattr(
        "src.chat.vectordb.get_by_metadata",
        lambda *args, **kwargs: [
            {"id": "c-i-parent", "text": "Art. 1 ...", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO I"}},
            {"id": "c-v-parent", "text": "Art. 33 ...", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO V"}},
        ],
    )

    out = chat._supplement_chapter_matches(initial, "colecao", "Resuma o capitulo 1")
    ids = [item["id"] for item in out]
    assert "c-i-parent" in ids


def test_expand_exact_chapter_context_adds_all_target_chapter_chunks(monkeypatch):
    initial = [
        {
            "id": "hit-cap-ii",
            "text": "Art. 2",
            "score": 0.7,
            "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO II", "chunk_index": 5},
        }
    ]
    monkeypatch.setattr(
        "src.chat.vectordb.get_by_metadata",
        lambda *args, **kwargs: [
            {"id": "hit-cap-ii", "text": "Art. 2", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO II", "chunk_index": 5}},
            {"id": "cap-ii-child-a", "text": "§1", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO II", "chunk_index": 6}},
            {"id": "cap-ii-child-b", "text": "§2", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO II", "chunk_index": 7}},
            {"id": "cap-v-other", "text": "Art. 29", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO V", "chunk_index": 70}},
        ],
    )
    out = chat._expand_exact_chapter_context(initial, "colecao", "Resuma o capitulo 2")
    ids = {item["id"] for item in out}
    assert "cap-ii-child-a" in ids
    assert "cap-ii-child-b" in ids
    assert "cap-v-other" not in ids


def test_prioritize_exact_chapter_matches_moves_target_to_front():
    items = [
        {"id": "cap-v", "text": "Art. 29", "score": 0.95, "metadata": {"capitulo": "CAPÍTULO V"}},
        {"id": "cap-ii", "text": "Art. 2", "score": 0.60, "metadata": {"capitulo": "CAPÍTULO II", "chunk_type": "parent"}},
        {"id": "cap-iii", "text": "Art. 7", "score": 0.80, "metadata": {"capitulo": "CAPÍTULO III"}},
    ]
    out = chat._prioritize_exact_chapter_matches(items, "Resuma o capitulo 2")
    assert out[0]["id"] == "cap-ii"


def test_boost_section_hint_compatibility_prioritizes_matching_hint():
    items = [
        {
            "id": "cap-v",
            "text": "Art. 29",
            "score": 0.85,
            "metadata": {"capitulo": "CAPÍTULO V", "pdf_section_hints": "CAPÍTULO V | Assembleia"},
        },
        {
            "id": "cap-ii",
            "text": "Art. 2",
            "score": 0.65,
            "metadata": {"capitulo": "CAPÍTULO II", "pdf_section_hints": "CAPÍTULO II | Objetivos sociais"},
        },
    ]
    out = chat._boost_section_hint_compatibility(items, "Resuma o capitulo 2", bonus=0.2)
    assert out[0]["id"] == "cap-ii"


def test_enforce_summary_structural_scope_keeps_only_requested_chapter():
    items = [
        {"id": "cap-v", "text": "Art. 29", "score": 0.9, "metadata": {"capitulo": "CAPÍTULO V"}},
        {"id": "cap-ii-a", "text": "Art. 2", "score": 0.6, "metadata": {"capitulo": "CAPÍTULO II"}},
        {"id": "cap-ii-b", "text": "§ 1", "score": 0.55, "metadata": {"capitulo": "CAPÍTULO II"}},
    ]
    scoped, ok = chat._enforce_summary_structural_scope(items, "Resuma o capitulo 2", min_hits=1)
    assert ok is True
    assert {i["id"] for i in scoped} == {"cap-ii-a", "cap-ii-b"}


def test_enforce_summary_structural_scope_keeps_only_requested_section():
    items = [
        {"id": "sec-i", "text": "Art. 1", "score": 0.9, "metadata": {"secao": "SECAO I - GERAL"}},
        {"id": "sec-iii-a", "text": "Art. 8", "score": 0.7, "metadata": {"secao": "SECAO III - PENALIDADES"}},
        {"id": "sec-iii-b", "text": "Art. 9", "score": 0.65, "metadata": {"secao": "SECAO III - PENALIDADES"}},
    ]
    scoped, ok = chat._enforce_summary_structural_scope(items, "Explique a secao III", min_hits=1)
    assert ok is True
    assert {i["id"] for i in scoped} == {"sec-iii-a", "sec-iii-b"}


def test_metadata_structural_scope_from_seed_uses_doc_metadata(monkeypatch):
    seed = [
        {"id": "v1", "text": "hit", "score": 0.6, "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO V"}},
    ]
    monkeypatch.setattr(
        "src.chat.vectordb.get_by_metadata",
        lambda *args, **kwargs: [
            {"id": "ii-1", "text": "Art. 2", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO II", "chunk_index": 5, "page_number": 2}},
            {"id": "ii-2", "text": "§ 1", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO II", "chunk_index": 6, "page_number": 2}},
            {"id": "v-1", "text": "Art. 29", "metadata": {"doc_id": "estatuto.pdf", "capitulo": "CAPÍTULO V", "chunk_index": 70, "page_number": 7}},
        ],
    )
    scoped = chat._metadata_structural_scope_from_seed(
        physical_collection="colecao",
        question="Resuma o capitulo 2",
        seed_results=seed,
    )
    ids = [item["id"] for item in scoped]
    assert ids == ["ii-1", "ii-2"]


def test_build_on_demand_summary_from_scope_returns_none_when_empty():
    result = chat._build_on_demand_summary_from_scope(
        question="Resuma o capitulo 2",
        scoped_results=[],
        request_id="req-1",
        workspace_id="default",
        collection="col",
    )
    assert result is None


def test_extract_structural_scope_from_artifact(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPÍTULO I - DAS CARACTERÍSTICAS"},
            {"block_type": "body", "text": "Art. 1º - A cooperativa tem sede em Porto Alegre."},
            {"block_type": "body", "text": "Art. 2º - O prazo de duração é indeterminado."},
            {"block_type": "section_header", "text": "CAPÍTULO II - DOS OBJETIVOS SOCIAIS"},
            {"block_type": "body", "text": "Art. 3º - A cooperativa objetiva promover o desenvolvimento."},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr(
        "src.summaries.generate_summary_from_scope_text",
        lambda **kwargs: type("Summary", (), {"to_dict": lambda self: {
            "node_id": "x",
            "node_type": "capitulo",
            "label": kwargs["label"],
            "path": kwargs["path"],
            "resumo_executivo": "Resumo do capítulo I",
            "resumo_juridico": "",
            "pontos_chave": [],
            "artigos_cobertos": [],
            "obrigacoes": [],
            "restricoes": [],
            "definicoes": [],
            "text_length": len(kwargs["text"]),
            "source_hash": "",
            "source_text_length": len(kwargs["text"]),
            "status": "fallback_only",
            "validation_errors": [],
            "generation_meta": {},
        }})(),
    )
    summary = chat._extract_structural_scope_from_artifact(
        question="Resuma o capitulo 1",
        collection="teste",
        workspace_id="default",
    )
    assert summary is not None
    assert "CAPÍTULO I" in summary["label"]


def test_answer_structure_first_counts_chapters(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO I - DAS CARACTERISTICAS", "page_number": 1},
            {"block_type": "body", "text": "Art. 1 - Texto.", "page_number": 1},
            {"block_type": "section_header", "text": "CAPITULO II - DOS OBJETIVOS SOCIAIS", "page_number": 2},
            {"block_type": "body", "text": "Art. 2 - Texto.", "page_number": 2},
            {"block_type": "section_header", "text": "CAPITULO III - DA ADMINISTRACAO", "page_number": 3},
            {"block_type": "body", "text": "Art. 3 - Texto.", "page_number": 3},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(collection="teste", question="Quantos capitulos tem no estatuto?", request_id="req-count")

    assert "3 capitulo(s)" in result.answer.lower()
    assert result.sources[0].metadata["requested_type"] == "capitulo"


def test_answer_structure_first_lists_articles_within_chapter(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO I - DAS CARACTERISTICAS", "page_number": 1},
            {"block_type": "body", "text": "Art. 1 - Texto.", "page_number": 1},
            {"block_type": "section_header", "text": "CAPITULO II - DOS OBJETIVOS SOCIAIS", "page_number": 2},
            {"block_type": "body", "text": "Art. 2 - Texto.", "page_number": 2},
            {"block_type": "body", "text": "Art. 3 - Texto.", "page_number": 2},
            {"block_type": "section_header", "text": "CAPITULO III - DA ADMINISTRACAO", "page_number": 3},
            {"block_type": "body", "text": "Art. 4 - Texto.", "page_number": 3},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(collection="teste", question="Quais artigos estao no capitulo II?", request_id="req-list")

    assert "art. 2" in result.answer.lower()
    assert "art. 3" in result.answer.lower()
    assert result.sources[0].metadata["scope"] == "CAPITULO II - DOS OBJETIVOS SOCIAIS"


def test_answer_structure_first_locates_last_chapter(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO I - DAS CARACTERISTICAS", "page_number": 1},
            {"block_type": "body", "text": "Art. 1 - Texto.", "page_number": 1},
            {"block_type": "section_header", "text": "CAPITULO II - DOS OBJETIVOS SOCIAIS", "page_number": 2},
            {"block_type": "body", "text": "Art. 2 - Texto.", "page_number": 2},
            {"block_type": "section_header", "text": "CAPITULO III - DA ADMINISTRACAO", "page_number": 5},
            {"block_type": "body", "text": "Art. 3 - Texto.", "page_number": 5},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(collection="teste", question="Qual e o ultimo capitulo?", request_id="req-locate")

    assert "capitulo iii - da administracao" in result.answer.lower()
    assert "pagina 5" in result.answer.lower()


def test_answer_structure_first_contains_section_in_chapter(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO V - DA ASSEMBLEIA GERAL", "page_number": 7},
            {"block_type": "section_header", "text": "SECAO I - DA ASSEMBLEIA GERAL ORDINARIA", "page_number": 7},
            {"block_type": "body", "text": "Art. 29 - Texto.", "page_number": 7},
            {"block_type": "section_header", "text": "SECAO III - DA ASSEMBLEIA GERAL EXTRAORDINARIA", "page_number": 8},
            {"block_type": "body", "text": "Art. 33 - Texto.", "page_number": 8},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(collection="teste", question="O capitulo V tem secao III?", request_id="req-contains")

    assert result.answer.lower().startswith("sim.")
    assert "secao iii" in result.answer.lower()
    assert result.sources[0].metadata["scope"] == "CAPITULO V - DA ASSEMBLEIA GERAL"


def test_answer_structure_first_contains_negative_article(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO II - DOS OBJETIVOS SOCIAIS", "page_number": 2},
            {"block_type": "body", "text": "Art. 2 - Texto.", "page_number": 2},
            {"block_type": "body", "text": "Art. 3 - Texto.", "page_number": 2},
            {"block_type": "section_header", "text": "CAPITULO III - DA ADMINISTRACAO", "page_number": 5},
            {"block_type": "body", "text": "Art. 4 - Texto.", "page_number": 5},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(collection="teste", question="Existe art. 41 no capitulo II?", request_id="req-contains-neg")

    assert result.answer.lower().startswith("nao.")
    assert "capitulo ii" in result.answer.lower()


def test_answer_exact_article_from_artifact_uses_requested_article(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO II - DOS OBJETIVOS SOCIAIS", "page_number": 2},
            {"block_type": "body", "text": "Art. 2º - A Cooperativa objetiva promover o desenvolvimento progressivo e a defesa de suas atividades de carater comum.", "page_number": 2},
            {"block_type": "body", "text": "Paragrafo unico - sem objetivo de lucro.", "page_number": 2},
            {"block_type": "body", "text": "Art. 3º - Poderao associar-se na Cooperativa os medicos que preencham os requisitos.", "page_number": 3},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    captured = {}

    def fake_chat(messages, system=""):
        captured["messages"] = messages
        return "O Art. 2 estabelece os objetivos sociais da Cooperativa."

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.chat", fake_chat)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(
        collection="teste",
        question="Quais sao os objetivos da Cooperativa previstos no Art. 2º?",
        request_id="req-art-2",
        domain_profile="legal",
    )

    assert "art. 2" in result.answer.lower()
    assert result.sources[0].metadata["artigo"] == "Art. 2"
    joined = "\n".join(msg["content"] for msg in captured["messages"])
    assert "Art. 2º" in joined or "Art. 2" in joined
    assert "Art. 3º" not in joined


def test_answer_exact_article_from_artifact_for_locate_excerpt(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {"block_type": "section_header", "text": "CAPITULO V - DA ASSEMBLEIA GERAL", "page_number": 7},
            {"block_type": "body", "text": "Art. 33 - E da competencia exclusiva da Assembleia Geral Extraordinaria deliberar sobre os seguintes assuntos.", "page_number": 8},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.chat", lambda messages, system="": "O Art. 33 trata da competencia exclusiva da Assembleia Geral Extraordinaria.")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(collection="teste", question="Art. 33", request_id="req-art-33")

    assert "art. 33" in result.answer.lower()
    assert result.sources[0].metadata["artigo"] == "Art. 33"


def test_answer_exact_article_from_header_block(monkeypatch, tmp_path):
    artifacts = tmp_path / "processed"
    artifacts.mkdir()
    payload = {
        "blocks": [
            {
                "block_type": "section_header",
                "text": "CAPITULO II DOS OBJETIVOS SOCIAIS Art. 2º - A sociedade objetiva, com base na colaboracao reciproca, promover o desenvolvimento progressivo e a defesa de suas atividades.",
                "page_number": 2,
            },
            {"block_type": "body", "text": "Paragrafo 1 - Sem objetivo de lucro.", "page_number": 2},
            {"block_type": "body", "text": "Art. 3º - Outro artigo.", "page_number": 3},
        ]
    }
    (artifacts / "estatuto.pdf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    class Doc:
        doc_id = "estatuto.pdf"
        filename = "estatuto.pdf"

    monkeypatch.setattr("src.chat.get_settings", lambda: _settings(pdf_pipeline_artifacts_dir=str(artifacts)))
    monkeypatch.setattr("src.controlplane.list_documents", lambda workspace_id, collection: [Doc()])
    monkeypatch.setattr("src.chat.llm.chat", lambda messages, system="": "O Art. 2 trata dos objetivos sociais da Cooperativa.")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(
        collection="teste",
        question="Quais sao os objetivos da Cooperativa previstos no Art. 2º?",
        request_id="req-art-header",
        domain_profile="legal",
    )

    assert "art. 2" in result.answer.lower()
    assert result.sources[0].metadata["artigo"] == "Art. 2"


def test_answer_table_first_aggregate_sum_by_state(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"nome": "Ana", "renda_mensal": "1000", "estado": "RJ", "cidade": "Rio de Janeiro"}),
        Rec(1, {"nome": "Bia", "renda_mensal": "2500", "estado": "RJ", "cidade": "Niteroi"}),
        Rec(2, {"nome": "Caio", "renda_mensal": "3000", "estado": "SP", "cidade": "Sao Paulo"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["nome", "renda_mensal", "estado", "cidade"])

    result = chat.answer(collection="cadastro", question="Qual e a renda do estado do Rio de Janeiro?", request_id="req-table-sum")

    assert "3.500,00" in result.answer
    assert "estado de RJ" in result.answer or "estado = RJ" in result.answer
    assert result.sources[0].metadata["source"] == "structured_store"
    assert result.sources[0].metadata["source_kind"] == "table_query"
    assert result.sources[0].metadata["plan"]["aggregation"] == "sum"
    assert "SELECT SUM(" in result.sources[0].metadata["query_summary"]
    assert "resultado:" in result.sources[0].metadata["result_preview"]


def test_answer_table_first_average_with_numeric_filter_and_state_preference(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"idade": "35", "renda_mensal": "1500.50", "estado": "SP", "cidade": "Sao Paulo"}),
        Rec(1, {"idade": "28", "renda_mensal": "2500.75", "estado": "SP", "cidade": "Campinas"}),
        Rec(2, {"idade": "45", "renda_mensal": "3500.00", "estado": "SP", "cidade": "Santos"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["idade", "renda_mensal", "estado", "cidade"])

    result = chat.answer(
        collection="cadastro",
        question="Qual e a media de renda em Sao Paulo com pessoas acima de 30 anos?",
        request_id="req-table-filtered-avg",
    )

    assert "2.500,25" in result.answer
    assert "estado de SP" in result.answer
    assert "cidade de Sao Paulo" not in result.answer
    sql = result.sources[0].metadata["query_summary"]
    assert "AVG(" in sql
    assert "estado" in sql
    assert "idade" in sql
    assert " > ?" in sql


def test_answer_table_first_groupby_average(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"renda_mensal": "1000", "estado": "RJ"}),
        Rec(1, {"renda_mensal": "3000", "estado": "RJ"}),
        Rec(2, {"renda_mensal": "4000", "estado": "SP"}),
        Rec(3, {"renda_mensal": "6000", "estado": "SP"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["renda_mensal", "estado"])

    result = chat.answer(collection="cadastro", question="Qual a media por estado?", request_id="req-table-group")

    assert "por estado" in result.answer.lower()
    assert "sp: r$ 5.000,00" in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "groupby"
    assert "resultado:" not in result.sources[0].metadata["result_preview"]


def test_answer_table_first_average_age_uses_years_not_currency(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"idade": "20", "estado": "CE"}),
        Rec(1, {"idade": "40", "estado": "CE"}),
        Rec(2, {"idade": "60", "estado": "SP"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["idade", "estado"])

    result = chat.answer(collection="cadastro", question="Qual e a media de idade dos clientes do Ceara?", request_id="req-table-age")

    assert "30,00 anos" in result.answer
    assert "R$" not in result.answer
    assert result.sources[0].metadata["plan"]["metric_unit"] == "anos"


def test_answer_table_first_count_formats_as_integer(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"id_cliente": "1", "estado": "RJ"}),
        Rec(1, {"id_cliente": "2", "estado": "RJ"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["id_cliente", "estado"])

    result = chat.answer(collection="cadastro", question="Quantos clientes tem no estado do Rio de Janeiro?", request_id="req-table-count")

    assert " 2." in result.answer or result.answer.endswith(" 2.")
    assert "2,00" not in result.answer
    assert result.sources[0].metadata["plan"]["aggregation"] == "count"


def test_answer_table_first_distinct_inventory(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"id_cliente": "1", "estado": "RJ"}),
        Rec(1, {"id_cliente": "2", "estado": "SP"}),
        Rec(2, {"id_cliente": "3", "estado": "BA"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["id_cliente", "estado"])

    result = chat.answer(collection="cadastro", question="Quais sao os estados que estao na base?", request_id="req-table-distinct")

    assert "estados presentes na base" in result.answer.lower()
    assert "BA" in result.answer
    assert "RJ" in result.answer
    assert "SP" in result.answer
    assert result.sources[0].metadata["plan"]["operation"] == "distinct"


def test_answer_table_first_uses_collection_context_hint(monkeypatch):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: True)

    captured = {}

    def fake_plan_query(collection, question, context_hint=""):
        captured["context_hint"] = context_hint
        return {
            "operation": "aggregate",
            "metric_column": None,
            "aggregation": "count",
            "filters": [{"column": "estado", "operator": "=", "value": "RJ"}],
            "group_by": [],
            "limit": 1,
            "assumption": "",
        }

    monkeypatch.setattr("src.structured_store.plan_query", fake_plan_query)
    monkeypatch.setattr("src.structured_store.execute_plan", lambda collection, plan: {"operation": "aggregate", "value": 7, "plan": plan})
    monkeypatch.setattr("src.chat._get_collection_context_hint", lambda workspace_id, collection: "Base de clientes com cidade, estado e renda mensal.")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    result = chat.answer(
        collection="cadastro",
        question="Quantos tem no estado do Rio de Janeiro?",
        request_id="req-context-hint",
    )

    assert captured["context_hint"].startswith("Base de clientes")
    assert "clientes" in result.answer.lower()
    assert result.sources[0].metadata["context_hint"].startswith("Base de clientes")


def test_answer_table_first_distinct_inventory_states_uf(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"id_cliente": "1", "estado": "RJ"}),
        Rec(1, {"id_cliente": "2", "estado": "SP"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["id_cliente", "estado"])

    result = chat.answer(collection="cadastro", question="Quais sao os estados UF que estao na tabela?", request_id="req-table-uf")

    assert "estados presentes na base" in result.answer.lower()
    assert "RJ" in result.answer
    assert "SP" in result.answer
    assert "id clientes" not in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "distinct"
    assert result.sources[0].metadata["plan"]["dimension_column"] == "estado"


def test_answer_table_first_schema_columns(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"id_cliente": "1", "nome": "Ana", "estado": "RJ"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["id_cliente", "nome", "estado"])

    result = chat.answer(collection="cadastro", question="Quais sao as colunas da tabela?", request_id="req-table-schema")

    assert "colunas da tabela" in result.answer.lower()
    assert "id_cliente" in result.answer
    assert "estado" in result.answer
    assert "dimensoes principais" in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "schema"


def test_answer_table_first_describe_column(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"id_cliente": "1", "score_credito": "780"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["id_cliente", "score_credito"])

    result = chat.answer(collection="cadastro", question="O que significa a coluna score_credito?", request_id="req-table-describe")

    assert "score_credito" in result.answer
    assert "tipo semantico" in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "describe_column"


def test_answer_table_first_compare_is_executive(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {"renda_mensal": "1000", "estado": "RJ"}),
        Rec(1, {"renda_mensal": "3000", "estado": "RJ"}),
        Rec(2, {"renda_mensal": "4000", "estado": "SP"}),
        Rec(3, {"renda_mensal": "6000", "estado": "SP"}),
    ]
    structured_store.upsert_records("cadastro", "doc.csv", records, ["renda_mensal", "estado"])

    result = chat.answer(collection="cadastro", question="Compare SP e RJ em renda media por estado", request_id="req-table-compare")

    assert "comparativo" in result.answer.lower()
    assert "lidera" in result.answer.lower()
    assert "diferenca" in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "compare"


def test_answer_table_first_catalog_lookup_by_id(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {
            "procedimento": "20017",
            "descricao_unimed_poa": "Visita Domiciliar",
            "cobertura_unimed_poa": "Hospitalar",
            "orientacao_autorizacao_call_center": "Necessita autorizacao da Unidade",
        }),
    ]
    structured_store.upsert_records(
        "rol",
        "doc.csv",
        records,
        ["procedimento", "descricao_unimed_poa", "cobertura_unimed_poa", "orientacao_autorizacao_call_center"],
    )

    monkeypatch.setattr(
        "src.chat._get_collection_context_hint",
        lambda workspace_id, collection: "Catalogo de procedimentos com cobertura e autorizacao.",
    )

    result = chat.answer(collection="rol", question="Qual e o procedimento com esse id 20017?", request_id="req-catalog-id")

    assert "codigo 20017" in result.answer.lower()
    assert "visita domiciliar" in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "catalog_lookup"
    assert result.sources[0].metadata["citation_label"] == "registro de catalogo"


def test_answer_table_first_catalog_field_lookup(monkeypatch, tmp_path):
    from src import structured_store

    db_path = tmp_path / "structured.duckdb"
    settings = _settings(structured_store_path=str(db_path), query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store._CONN", None)
    monkeypatch.setattr("src.structured_store._BACKEND", "")
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    class Rec:
        def __init__(self, row_index, fields):
            self.row_index = row_index
            self.page_number = None
            self.fields = fields
            self.raw_row = ""
            self.texto_canonico = "; ".join(f"{k}: {v}" for k, v in fields.items())

    records = [
        Rec(0, {
            "procedimento": "20017",
            "descricao_unimed_poa": "Visita Domiciliar",
            "cobertura_unimed_poa": "Hospitalar",
            "orientacao_autorizacao_call_center": "Necessita autorizacao da Unidade",
        }),
    ]
    structured_store.upsert_records(
        "rol",
        "doc.csv",
        records,
        ["procedimento", "descricao_unimed_poa", "cobertura_unimed_poa", "orientacao_autorizacao_call_center"],
    )

    monkeypatch.setattr(
        "src.chat._get_collection_context_hint",
        lambda workspace_id, collection: "Catalogo de procedimentos com cobertura e autorizacao.",
    )

    result = chat.answer(collection="rol", question="Qual a cobertura do 20017?", request_id="req-catalog-field")

    assert "hospitalar" in result.answer.lower()
    assert "procedimento 20017" in result.answer.lower()
    assert result.sources[0].metadata["plan"]["operation"] == "catalog_field_lookup"


def test_answer_catalog_code_from_artifact_when_no_structured_store(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "rol.json"
    tabular_artifact_path = tmp_path / "rol.tabular.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "20101325 Consulta em pronto atendimento", "page_number": 1},
                    {"text": "Cobertura Hospitalar", "page_number": 1},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tabular_artifact_path.write_text(
        json.dumps(
            {
                "doc_id": "rol",
                "collection": "rol",
                "table_type": "catalog",
                "column_names": ["procedimento", "descricao"],
                "records": [
                    {
                        "row_index": 0,
                        "page_number": 1,
                        "fields": {"procedimento": "20101325", "descricao": "Consulta em pronto atendimento"},
                        "texto_canonico": "procedimento: 20101325; descricao: Consulta em pronto atendimento",
                        "raw_row": "20101325 | Consulta em pronto atendimento",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.chat._resolve_processed_json",
        lambda doc_id, filename: artifact_path,
    )
    monkeypatch.setattr(
        "src.chat._resolve_processed_tabular_json",
        lambda doc_id, filename: tabular_artifact_path,
    )
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "rol", "filename": "rol.pdf"})()],
    )

    result = chat.answer(collection="rol", question="Qual e o procedimento 20101325?", request_id="req-artifact-catalog")

    assert "20101325" in result.answer
    assert "pronto atendimento" in result.answer.lower()
    assert result.sources[0].metadata["source_kind"] == "artifact_lookup"
    assert result.sources[0].metadata["artifact_record"]["descricao"] == "Consulta em pronto atendimento"


def test_answer_catalog_field_from_artifact_uses_catalog_schema(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "rol.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento\nDescrição Unimed POA\nCobertura\nPrazo Autorização\nOrientação Autorização"},
                    {"text": "10102914"},
                    {"text": "Atendimento / Acompanhamento em Hospital-dia Psiquiátrico"},
                    {"text": "Somente Regulamentados\nHospitalar\nNÃO\nHospital-dia - até 5 (cinco) dias úteis"},
                    {"text": "Necessita autorização Auditoria Médica e Unidade"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.chat._resolve_processed_json",
        lambda doc_id, filename: artifact_path,
    )
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "rol", "filename": "rol.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {
                "column_name": "procedimento",
                "semantic_type": "identifier",
                "aliases": ["procedimento", "codigo", "id"],
            },
            {
                "column_name": "prazo_autorizacao_conforme_rn_n_623_ans",
                "semantic_type": "deadline_rule",
                "aliases": ["prazo", "prazo autorizacao"],
            },
        ],
    )

    result = chat.answer(
        collection="rol",
        question="Qual o Prazo Autorizacao do procedimento 10102914?",
        request_id="req-artifact-field",
    )

    assert "prazo de autorização" in result.answer.lower()
    assert "5 (cinco) dias úteis" in result.answer.lower()
    assert result.sources[0].metadata["target_column"] in {
        "prazo_autorizacao_conforme_rn_n_623_ans",
        "prazo_autorizacao_conforme_rn_no_623_ans",
    }


def test_answer_generic_catalog_field_from_artifact_uses_semantic_profiles(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalogo.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Codigo do Item Nome do Item Categoria Prazo de Entrega Regra de Liberacao"},
                    {"text": "50123"},
                    {"text": "Kit Especial de Internacao"},
                    {"text": "Hospitalar\nAte 72 horas\nLiberacao da central tecnica"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "codigo_item", "semantic_type": "identifier", "role": "identifier", "aliases": ["codigo", "item"]},
            {"column_name": "nome_item", "semantic_type": "catalog_title", "role": "text", "aliases": ["nome do item", "nome item"]},
            {"column_name": "categoria_item", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["categoria"]},
            {"column_name": "prazo_entrega", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo de entrega", "prazo"]},
            {"column_name": "regra_liberacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["regra de liberacao", "liberacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual o prazo do item 50123?",
        request_id="req-generic-catalog-artifact",
    )

    assert "50123" in result.answer
    assert "72 horas" in result.answer.lower()
    assert result.sources[0].metadata["target_column"] == "prazo_entrega"


def test_answer_catalog_field_request_prioritizes_deadline_over_authorization(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "rol.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Prazo Autorizacao", "page_number": 1},
                    {"text": "10102914", "page_number": 1},
                    {"text": "Atendimento / Acompanhamento em Hospital-dia Psiquiatrico", "page_number": 1},
                    {"text": "Hospitalar\nAte 5 (cinco) dias uteis", "page_number": 1},
                    {"text": "Necessita autorizacao Auditoria Medica e Unidade", "page_number": 1},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "rol", "filename": "rol.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "prazo_autorizacao", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "orientacao_autorizacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["autorizacao", "orientacao autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="rol",
        question="Qual o Prazo Autorizacao do procedimento 10102914?",
        request_id="req-catalog-deadline-priority",
    )

    assert "prazo de autorização" in result.answer.lower()
    assert "5 (cinco) dias uteis" in result.answer.lower()
    assert "autorizacao" in result.answer.lower()
    assert result.sources[0].metadata["target_column"] == "prazo_autorizacao"


def test_answer_catalog_deadline_from_realistic_split_blocks(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    blocks = [{"text": f"bloco {i}", "page_number": 3} for i in range(58)]
    blocks.extend(
        [
            {
                "text": "Procedimento Descricao Unimed POA Segmentacao ANS Cobertura Unimed POA Emergencia Prazo Autorizacao conforme RN 623 ANS Orientacao Autorizacao Call Center",
                "page_number": 3,
            },
            {"text": "10102914", "page_number": 3},
            {
                "text": "Atendimento / Acompanhamento em Hospital-dia Psiquiatrico (com diretriz definida pela ANS - n 109)",
                "page_number": 3,
            },
            {
                "text": "Somente\nRegulamentados\nHospitalar\nNAO\nHospital-dia - ate 5 (cinco) dias uteis",
                "page_number": 3,
            },
            {
                "text": "Necessita autorizacao Auditoria Medica e Unidade",
                "page_number": 3,
            },
            {"text": "10103015", "page_number": 3},
        ]
    )
    artifact_path = tmp_path / "rol_realistic.json"
    artifact_path.write_text(json.dumps({"blocks": blocks}, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "rol", "filename": "rol.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "segmentacao_ans", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["segmentacao ans"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "emergencia", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_n_623_ans", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "orientacao_autorizacao_call_center", "semantic_type": "authorization_rule", "role": "text", "aliases": ["orientacao autorizacao", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="rol",
        question="Qual o Prazo Autorizacao do procedimento 10102914?",
        request_id="req-catalog-realistic-deadline",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "prazo de autorizacao" in normalized_answer
    assert "hospital-dia - ate 5 (cinco) dias uteis" in normalized_answer
    assert "auditoria medica e unidade" in result.answer.lower()
    assert result.sources[0].metadata["target_column"] in {
        "prazo_autorizacao_conforme_rn_n_623_ans",
        "prazo_autorizacao_conforme_rn_no_623_ans",
    }


def test_answer_catalog_field_from_artifact_uses_silver_json_prompt(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "rol_prompt.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Prazo Autorizacao", "page_number": 1},
                    {"text": "10102914", "page_number": 1},
                    {"text": "Atendimento em Hospital-dia Psiquiatrico", "page_number": 1},
                    {"text": "Hospitalar\nAte 5 (cinco) dias uteis", "page_number": 1},
                    {"text": "Necessita autorizacao Auditoria Medica e Unidade", "page_number": 1},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_chat(messages, system=""):
        captured["system"] = system
        captured["messages"] = messages
        return "O prazo de autorizacao do procedimento 10102914 e: Ate 5 (cinco) dias uteis."

    monkeypatch.setattr("src.chat.llm.chat", fake_chat)
    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "rol", "filename": "rol.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "prazo_autorizacao", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "orientacao_autorizacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="rol",
        question="Qual o Prazo Autorizacao do procedimento 10102914?",
        request_id="req-silver-json-prompt",
    )

    joined = "\n".join(msg["content"] for msg in captured["messages"])
    assert "prazo_autorizacao" in joined
    assert "\"10102914\"" in joined
    assert "\"orientacao_autorizacao\"" in joined
    assert "registro estruturado da camada prata" in joined.lower()
    assert "ate 5 (cinco) dias uteis" in chat._normalize_for_match(result.answer)


def test_catalog_artifact_parser_keeps_rows_independent_by_code_boundary(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_boundary.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Prazo Autorizacao", "page_number": 1},
                    {"text": "10001", "page_number": 1},
                    {"text": "Primeiro item", "page_number": 1},
                    {"text": "Ate 2 dias uteis", "page_number": 1},
                    {"text": "Nao necessita autorizacao", "page_number": 1},
                    {"text": "10002", "page_number": 1},
                    {"text": "Segundo item", "page_number": 1},
                    {"text": "Ate 7 dias uteis", "page_number": 1},
                    {"text": "Necessita autorizacao da unidade", "page_number": 1},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "prazo", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo"]},
            {"column_name": "autorizacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual o prazo do procedimento 10002?",
        request_id="req-catalog-boundary",
    )

    assert "10002" in result.answer
    assert "7 dias uteis" in result.answer.lower()
    assert "2 dias uteis" not in result.answer.lower()


def test_catalog_artifact_parser_handles_header_split_across_blocks(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_split_header.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Unimed POA Cobertura Unimed POA", "page_number": 1},
                    {"text": "Segmentacao ANS Emergencia Prazo Autorizacao conforme RN 623 ANS", "page_number": 1},
                    {"text": "Orientacao Autorizacao Call Center", "page_number": 1},
                    {"text": "10102914", "page_number": 1},
                    {"text": "Atendimento em Hospital-dia Psiquiatrico", "page_number": 1},
                    {"text": "Somente Regulamentados\nHospitalar\nNAO\nAte 5 (cinco) dias uteis", "page_number": 1},
                    {"text": "Necessita autorizacao Auditoria Medica e Unidade", "page_number": 1},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "segmentacao_ans", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["segmentacao ans"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "emergencia", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_n_623_ans", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "orientacao_autorizacao_call_center", "semantic_type": "authorization_rule", "role": "text", "aliases": ["orientacao autorizacao", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual o prazo do procedimento 10102914?",
        request_id="req-catalog-split-header",
    )

    assert "5 (cinco) dias uteis" in result.answer.lower()
    assert result.sources[0].metadata["target_column"] in {
        "prazo_autorizacao_conforme_rn_n_623_ans",
        "prazo_autorizacao_conforme_rn_no_623_ans",
    }


def test_catalog_artifact_parser_skips_repeated_header_on_next_page(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    repeated_header = "Procedimento Descricao Unimed POA Segmentacao ANS Cobertura Unimed POA Emergencia Prazo Autorizacao Orientacao Autorizacao"
    artifact_path = tmp_path / "catalog_repeated_header.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": repeated_header, "page_number": 1},
                    {"text": "10102914", "page_number": 1},
                    {"text": "Atendimento em Hospital-dia Psiquiatrico", "page_number": 1},
                    {"text": repeated_header, "page_number": 2},
                    {"text": "Somente Regulamentados\nHospitalar\nNAO\nAte 5 (cinco) dias uteis", "page_number": 2},
                    {"text": "Necessita autorizacao Auditoria Medica e Unidade", "page_number": 2},
                    {"text": "10103015", "page_number": 2},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "segmentacao_ans", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["segmentacao ans"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "emergencia", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_n_623_ans", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "orientacao_autorizacao_call_center", "semantic_type": "authorization_rule", "role": "text", "aliases": ["orientacao autorizacao", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual o prazo do procedimento 10102914?",
        request_id="req-catalog-repeated-header",
    )

    assert "5 (cinco) dias uteis" in result.answer.lower()
    assert "procedimento descricao" not in result.answer.lower()


def test_catalog_artifact_parser_handles_record_crossing_pages(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_cross_page.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Prazo Autorizacao", "page_number": 1},
                    {"text": "10102914", "page_number": 1},
                    {"text": "Atendimento em Hospital-dia Psiquiatrico", "page_number": 1},
                    {"text": "Procedimento Descricao Prazo Autorizacao", "page_number": 2},
                    {"text": "Hospitalar\nAte 5 (cinco) dias uteis", "page_number": 2},
                    {"text": "Necessita autorizacao Auditoria Medica e Unidade", "page_number": 2},
                    {"text": "10103015", "page_number": 2},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "cobertura", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura"]},
            {"column_name": "prazo", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo"]},
            {"column_name": "autorizacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual o prazo do procedimento 10102914?",
        request_id="req-catalog-cross-page",
    )

    assert "5 (cinco) dias uteis" in result.answer.lower()
    assert "10103015" not in result.answer


def test_catalog_artifact_field_lookup_prefers_exact_code_over_similar_code(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_similar_codes.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Cobertura Emergencia Prazo Autorizacao Orientacao Autorizacao", "page_number": 1},
                    {"text": "10101063", "page_number": 1},
                    {"text": "Teleconsulta Eletiva", "page_number": 1},
                    {"text": "Ambas (REG e Nao REG)\nAmbulatorial\nNAO\nConsultas nao autorizam", "page_number": 1},
                    {"text": "Unimax e Unipart nao necessita autorizacao, Unifacil autorizacao obrigatoria", "page_number": 1},
                    {"text": "10106103", "page_number": 1},
                    {"text": "Pericia medica", "page_number": 1},
                    {"text": "Ambas (REG e Nao REG)\nAmbulatorial\nNAO\nAte 3 dias uteis", "page_number": 1},
                    {"text": "Necessita autorizacao", "page_number": 1},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "cobertura", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura"]},
            {"column_name": "emergencia", "semantic_type": "flag_boolean", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "orientacao_autorizacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["orientacao autorizacao", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual e o prazo de autorizacao do id 10106103?",
        request_id="req-catalog-similar-code",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "10106103" in result.answer
    assert "pericia medica" in normalized_answer
    assert "ate 3 dias uteis" in normalized_answer
    assert "teleconsulta eletiva" not in normalized_answer
    assert result.sources[0].metadata["page_number"] == 1


def test_catalog_artifact_field_lookup_returns_safe_message_when_field_missing(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_missing_field.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Cobertura", "page_number": 2},
                    {"text": "10101063", "page_number": 2},
                    {"text": "Teleconsulta Eletiva", "page_number": 2},
                    {"text": "Ambulatorial", "page_number": 2},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "prazo_autorizacao", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual e o prazo de autorizacao do id 10101063?",
        request_id="req-catalog-missing-field",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "nao localizei o campo prazo" in normalized_answer
    assert "10101063" in result.answer
    assert result.sources[0].metadata["page_number"] == 2


def test_catalog_field_request_prefers_deadline_column_over_prazo_urgencia(monkeypatch):
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "authorization_rule", "aliases": ["prazo autorizacao", "prazo"]},
            {"column_name": "prazo_dias", "semantic_type": "deadline_rule", "aliases": ["prazo dias", "prazo"]},
            {"column_name": "prazo_faixa", "semantic_type": "deadline_rule", "aliases": ["prazo faixa", "prazo"]},
            {"column_name": "prazo_urgencia", "semantic_type": "deadline_rule", "aliases": ["prazo urgencia", "urgencia", "prazo"]},
        ],
    )

    result = chat._catalog_field_request("Qual e o prazo de autorizacao do id 10108009?", "default", "rol")

    assert result is not None
    assert result[0] != "prazo_urgencia"
    assert result[0] in {"prazo_autorizacao_conforme_rn_no_623_ans", "prazo_dias"}


def test_catalog_field_request_prefers_raw_authorization_deadline_over_days(monkeypatch):
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "prazo_dias", "semantic_type": "deadline_rule", "aliases": ["dias", "prazo dias", "prazo"]},
            {"column_name": "prazo_urgencia", "semantic_type": "deadline_rule", "aliases": ["urgencia", "prazo urgencia", "prazo"]},
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "authorization_rule", "aliases": ["prazo autorizacao", "prazo", "autorizacao"]},
            {"column_name": "orientacao_autorizacao_call_center", "semantic_type": "authorization_rule", "aliases": ["orientacao autorizacao", "autorizacao"]},
        ],
    )

    result = chat._catalog_field_request("Qual e o prazo de autorizacao do id 20101163?", "default", "rol")

    assert result is not None
    assert result[0] == "prazo_autorizacao_conforme_rn_no_623_ans"


def test_catalog_artifact_parser_can_fallback_to_global_header_when_local_header_missing(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    blocks = [
        {"text": "Procedimento Descricao Unimed POA Cobertura Unimed POA Segmentacao ANS Emergencia Prazo Autorizacao conforme RN 623 ANS Orientacao Autorizacao Call Center", "page_number": 1},
    ]
    for i in range(140):
        blocks.append({"text": f"bloco filler {i}", "page_number": 3})
    blocks.extend(
        [
            {"text": "10108009", "page_number": 6},
            {"text": "Reavaliacao Plano Terapeutico para Terapias dos Transtornos Globais do Desenvolvimento - indicada a cada 6 meses.", "page_number": 6},
            {"text": "Somente Regulamentados\nAmbulatorial\nNAO\nConsultas/sessoes - ate 5 (cinco) dias uteis", "page_number": 6},
            {"text": "10201017", "page_number": 6},
        ]
    )
    artifact_path = tmp_path / "catalog_global_header.json"
    artifact_path.write_text(json.dumps({"blocks": blocks}, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "segmentacao_ans", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["segmentacao ans"]},
            {"column_name": "emergencia", "semantic_type": "flag_boolean", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "deadline_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual e o prazo de autorizacao do id 10108009?",
        request_id="req-catalog-global-header",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "10108009" in result.answer
    assert "5 (cinco) dias uteis" in normalized_answer
    assert result.sources[0].metadata["target_column"] == "prazo_autorizacao_conforme_rn_no_623_ans"
    assert result.sources[0].metadata["page_number"] == 6


def test_catalog_artifact_unavailable_deadline_mentions_sem_cobertura(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_sem_cobertura.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Cobertura", "page_number": 7},
                    {"text": "20101163", "page_number": 7},
                    {"text": "Pulsoterapia (por Sessao)", "page_number": 7},
                    {"text": "Sem Cobertura\nSem Cobertura", "page_number": 7},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao"]},
            {"column_name": "cobertura", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura"]},
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "authorization_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual e o prazo de autorizacao do id 20101163?",
        request_id="req-catalog-sem-cobertura",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "20101163" in result.answer
    assert "sem cobertura" in normalized_answer
    assert "nao ha prazo" in normalized_answer
    assert "autorizacao aplicavel" in normalized_answer
    assert result.sources[0].metadata["page_number"] == 7


def test_catalog_artifact_single_block_record_with_sem_cobertura_is_not_folded_into_description(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_single_block_sem_cobertura.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Unimed POA Cobertura Unimed POA Segmentacao ANS Emergencia Prazo Autorizacao conforme RN 623 ANS Orientacao Autorizacao", "page_number": 1},
                    {"text": "20101163\nPulsoterapia (por Sessao)\nSem Cobertura\nSem Cobertura", "page_number": 7},
                    {"text": "20101171\nOutro procedimento", "page_number": 7},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "segmentacao_ans", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["segmentacao ans"]},
            {"column_name": "emergencia", "semantic_type": "flag_boolean", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "authorization_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual e o prazo de autorizacao do id 20101163?",
        request_id="req-catalog-single-block-sem-cobertura",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "pulsoterapia (por sessao)" in normalized_answer
    assert "sem cobertura" in normalized_answer
    assert "nao ha prazo" in normalized_answer


def test_catalog_field_answer_aggregates_related_row_information(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_aggregate_context.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "Procedimento Descricao Unimed POA Cobertura Unimed POA Segmentacao ANS Emergencia Prazo Autorizacao conforme RN 623 ANS Orientacao Autorizacao", "page_number": 6},
                    {"text": "10108009", "page_number": 6},
                    {"text": "Reavaliacao Plano Terapeutico para Terapias dos Transtornos Globais do Desenvolvimento - indicada a cada 6 meses.", "page_number": 6},
                    {"text": "Somente Regulamentados\nAmbulatorial\nNAO\nConsultas/sessoes - ate 5 (cinco) dias uteis", "page_number": 6},
                    {"text": "Nao necessita autorizacao", "page_number": 6},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "segmentacao_ans", "semantic_type": "catalog_attribute", "role": "text", "aliases": ["segmentacao ans"]},
            {"column_name": "emergencia", "semantic_type": "flag_boolean", "role": "text", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "authorization_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo", "autorizacao"]},
            {"column_name": "orientacao_autorizacao", "semantic_type": "authorization_rule", "role": "text", "aliases": ["orientacao autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual e o prazo de autorizacao do id 10108009?",
        request_id="req-catalog-aggregate-context",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "campos relacionados:" in normalized_answer
    assert "5 (cinco) dias uteis" in normalized_answer
    assert "cobertura:" in normalized_answer
    assert "segmentacao:" in normalized_answer
    assert "emergencia:" in normalized_answer


def test_catalog_artifact_parses_split_medical_row_with_deadline_and_authorization(monkeypatch, tmp_path):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: False)
    monkeypatch.setattr("src.chat.llm.embed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("vector path should not be used")))

    artifact_path = tmp_path / "catalog_split_medical_row.json"
    artifact_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {"text": "30711029\nImobilização de membro inferior\nAmbas (REG e Não REG)", "page_number": 104},
                    {"text": "Ambulatorial e", "page_number": 104},
                    {"text": "Hospitalar\nSIM\nTerapia em regime ambulatorial -", "page_number": 104},
                    {"text": "até 5 (cinco) dias úteis", "page_number": 104},
                    {"text": "Necessita autorização da", "page_number": 104},
                    {"text": "Unidade - OBS.: Não necessita autorização (Gerar", "page_number": 104},
                    {"text": "AIH) quando realizado em", "page_number": 104},
                    {"text": "consultório e emergência", "page_number": 104},
                    {"text": "30711037\nOutro procedimento", "page_number": 104},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.chat._resolve_processed_json", lambda doc_id, filename: artifact_path)
    monkeypatch.setattr(
        "src.controlplane.list_documents",
        lambda workspace_id, collection: [type("Doc", (), {"doc_id": "catalogo", "filename": "catalogo.pdf"})()],
    )
    monkeypatch.setattr(
        "src.controlplane.list_column_profiles",
        lambda workspace_id, collection: [
            {"column_name": "procedimento", "semantic_type": "identifier", "role": "identifier", "aliases": ["procedimento", "codigo", "id"]},
            {"column_name": "descricao_unimed_poa", "semantic_type": "catalog_title", "role": "text", "aliases": ["descricao unimed poa", "descricao"]},
            {"column_name": "cobertura_unimed_poa", "semantic_type": "coverage_rule", "role": "text", "aliases": ["cobertura unimed poa", "cobertura"]},
            {"column_name": "segmentacao_ans", "semantic_type": "dimension_category", "role": "dimension", "aliases": ["segmentacao ans", "segmentacao"]},
            {"column_name": "emergencia", "semantic_type": "flag_boolean", "role": "dimension", "aliases": ["emergencia"]},
            {"column_name": "prazo_autorizacao_conforme_rn_no_623_ans", "semantic_type": "authorization_rule", "role": "text", "aliases": ["prazo autorizacao", "prazo", "autorizacao"]},
            {"column_name": "orientacao_autorizacao_call_center", "semantic_type": "authorization_rule", "role": "text", "aliases": ["orientacao autorizacao", "autorizacao"]},
        ],
    )

    result = chat.answer(
        collection="catalogo",
        question="Qual é o prazo de autorização do id 30711029?",
        request_id="req-catalog-split-medical-row",
    )

    normalized_answer = chat._normalize_for_match(result.answer)
    assert "ate 5 (cinco) dias uteis" in normalized_answer
    assert "cobertura: ambas" in normalized_answer
    assert "segmentacao: ambulatorial e hospitalar" in normalized_answer
    assert "emergencia: sim" in normalized_answer
    assert "autorizacao: necessita autorizacao da" in normalized_answer


def test_answer_returns_direct_message_when_no_results(monkeypatch):
    monkeypatch.setattr("src.chat.get_settings", lambda: _settings())
    monkeypatch.setattr("src.chat.vectordb.resolve_query_collection", lambda c, m, workspace_id="default": f"{workspace_id}::{c}::{m}")
    monkeypatch.setattr("src.chat.llm.embed", lambda texts, model_name=None: [[0.1, 0.2]])
    monkeypatch.setattr("src.chat.vectordb.query", lambda *args, **kwargs: [])

    called = {"llm": False}

    def fake_chat(messages, system=""):
        called["llm"] = True
        return "nao deveria chamar"

    monkeypatch.setattr("src.chat.llm.chat", fake_chat)

    result = chat.answer(collection="geral", question="pergunta sem documentos", history=[], request_id="req-1")

    assert result.sources == []
    assert "encontrei" in result.answer.lower()
    assert "documentos" in result.answer.lower()
    assert called["llm"] is False


def test_answer_passes_requested_domain_profile_to_prompt(monkeypatch):
    monkeypatch.setattr("src.chat.get_settings", lambda: _settings())
    monkeypatch.setattr("src.chat.vectordb.resolve_query_collection", lambda c, m, workspace_id="default": f"{workspace_id}::{c}::{m}")
    monkeypatch.setattr("src.chat.llm.embed", lambda texts, model_name=None: [[0.1, 0.2]])
    monkeypatch.setattr(
        "src.chat.vectordb.query",
        lambda *args, **kwargs: [{"id": "chunk-1", "text": "Clausula de vigencia do contrato", "metadata": {"doc_id": "contrato.pdf"}, "distance": 0.1}],
    )

    captured = {}

    def fake_chat(messages, system=""):
        captured["system"] = system
        return "Resposta"

    monkeypatch.setattr("src.chat.llm.chat", fake_chat)

    result = chat.answer(collection="geral", question="Qual a clausula de vigencia?", domain_profile="legal")

    assert "legal" in captured["system"].lower()


# --- _expand_legal_context ---

def test_expand_legal_context_fetches_parents(monkeypatch):
    results = [
        {
            "id": "child-1",
            "text": "§ 1 Paragrafo do artigo.",
            "metadata": {"chunk_type": "child", "parent_key": "doc::Art. 41"},
            "score": 0.8,
        },
    ]

    def fake_get(col, where, include=None):
        conditions = where.get("$and", [])
        is_parent = any(
            c.get("chunk_type", {}).get("$eq") == "parent" for c in conditions if isinstance(c.get("chunk_type"), dict)
        )
        if is_parent:
            return [{"id": "parent-41", "text": "Art. 41 Texto completo.", "metadata": {"chunk_type": "parent", "parent_key": "doc::Art. 41"}}]
        return []

    monkeypatch.setattr("src.chat.vectordb.get_by_metadata", fake_get)

    expanded = chat._expand_legal_context(results, "test-collection")
    ids = [r["id"] for r in expanded]
    assert "child-1" in ids
    assert "parent-41" in ids
    assert len(expanded) == 2


def test_expand_legal_context_no_children():
    results = [
        {
            "id": "parent-1",
            "text": "Art. 1° Texto.",
            "metadata": {"chunk_type": "parent", "parent_key": "doc::Art. 1"},
            "score": 0.9,
        },
    ]
    expanded = chat._expand_legal_context(results, "test-collection")
    assert len(expanded) == 1


def test_trim_to_budget_keeps_all_when_under_limit():
    items = [
        {"id": "c1", "text": "curto", "score": 0.9},
        {"id": "c2", "text": "curto tb", "score": 0.5},
    ]
    result = chat._trim_to_budget(items, max_tokens=1000)
    assert len(result) == 2


def test_trim_to_budget_drops_lowest_score():
    items = [
        {"id": "c1", "text": "a" * 1000, "score": 0.9},
        {"id": "c2", "text": "b" * 1000, "score": 0.1},
    ]
    result = chat._trim_to_budget(items, max_tokens=400, chars_per_token=3.5)
    assert len(result) == 1
    assert result[0]["id"] == "c1"


def test_rerank_structural_continuity_prioritizes_heading_match():
    results = [
        {
            "id": "c1",
            "text": "Trecho generico.",
            "score": 0.2,
            "metadata": {"doc_id": "a.pdf"},
        },
        {
            "id": "c2",
            "text": "Trecho da secao correta.",
            "score": 0.15,
            "metadata": {"doc_id": "a.pdf", "secao": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS"},
        },
    ]
    reranked = chat._rerank_structural_continuity(results, "Quais sao os deveres do cooperado?", bonus=0.12)
    assert reranked[0]["id"] == "c2"


def test_expand_adjacent_structural_context_fetches_neighbors(monkeypatch):
    results = [
        {
            "id": "c2",
            "text": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS",
            "score": 0.4,
            "metadata": {"doc_id": "estatuto.pdf", "chunk_index": 1, "section": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS"},
        },
    ]

    monkeypatch.setattr(
        "src.chat.vectordb.get_by_metadata",
        lambda *args, **kwargs: [
            {"id": "c1", "text": "Art. 1 texto.", "metadata": {"doc_id": "estatuto.pdf", "chunk_index": 0}},
            {"id": "c2", "text": "Secao alvo.", "metadata": {"doc_id": "estatuto.pdf", "chunk_index": 1, "section": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS"}},
            {"id": "c3", "text": "Art. 2 texto.", "metadata": {"doc_id": "estatuto.pdf", "chunk_index": 2}},
        ],
    )

    expanded = chat._expand_adjacent_structural_context(
        results,
        "colecao",
        "Quais sao os deveres do cooperado?",
        window=1,
    )
    ids = [item["id"] for item in expanded]
    assert "c1" in ids
    assert "c2" in ids
    assert "c3" in ids


def test_mark_possible_contradiction_appends_warning():
    results = [
        {
            "id": "c1",
            "text": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS",
            "score": 0.4,
            "metadata": {"doc_id": "estatuto.pdf", "section": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS"},
        },
    ]
    answer = chat._mark_possible_contradiction(
        "Nao encontrei uma lista explicita dos deveres.",
        results,
        "Quais sao os deveres do cooperado?",
    )
    assert "Observacao:" in answer


def test_answer_regression_flags_partial_context_for_cooperado_duties(monkeypatch):
    monkeypatch.setattr(
        "src.chat.get_settings",
        lambda: _settings(
            retrieval_top_k=3,
            retrieval_structural_bonus=0.12,
            retrieval_adjacency_window=1,
        ),
    )
    monkeypatch.setattr(
        "src.chat.vectordb.resolve_query_collection",
        lambda c, m, workspace_id="default": f"{workspace_id}::{c}::{m}",
    )
    monkeypatch.setattr("src.chat.llm.embed", lambda texts, model_name=None: [[0.1, 0.2]])
    monkeypatch.setattr(
        "src.chat.vectordb.query",
        lambda *args, **kwargs: [
            {
                "id": "secao-1",
                "text": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS",
                "metadata": {
                    "doc_id": "estatuto.pdf",
                    "chunk_index": 1,
                    "section": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS",
                    "page_number": 2,
                },
                "distance": 0.1,
            }
        ],
    )

    def fake_get_by_metadata(collection_name, where, include=None):
        if where == {"doc_id": {"$eq": "estatuto.pdf"}}:
            return [
                {
                    "id": "art-9",
                    "text": "Art. 9. Sao direitos do cooperado: I - votar; II - participar.",
                    "metadata": {
                        "doc_id": "estatuto.pdf",
                        "chunk_index": 0,
                        "artigo": "Art. 9",
                        "page_number": 2,
                    },
                },
                {
                    "id": "secao-1",
                    "text": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS",
                    "metadata": {
                        "doc_id": "estatuto.pdf",
                        "chunk_index": 1,
                        "section": "SECAO II - DIREITOS E DEVERES DOS COOPERADOS",
                        "page_number": 2,
                    },
                },
                {
                    "id": "art-10",
                    "text": "Art. 10. Sao deveres do cooperado: I - cumprir o estatuto; II - integralizar quotas.",
                    "metadata": {
                        "doc_id": "estatuto.pdf",
                        "chunk_index": 2,
                        "artigo": "Art. 10",
                        "page_number": 2,
                    },
                },
            ]
        return []

    monkeypatch.setattr("src.chat.vectordb.get_by_metadata", fake_get_by_metadata)
    monkeypatch.setattr(
        "src.chat.llm.chat",
        lambda messages, system="": "Nao encontrei uma lista explicita dos deveres do cooperado.",
    )

    result = chat.answer(
        collection="geral",
        question="Quais sao os deveres do cooperado?",
        domain_profile="legal",
    )

    source_ids = [src.chunk_id for src in result.sources]
    assert "art-10" in source_ids
    assert "Observacao:" in result.answer


def test_retrieve_with_routing_structured_path(monkeypatch):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.chat.detect_injection", lambda q: None)
    monkeypatch.setattr("src.chat.sanitize_question", lambda q: q)
    monkeypatch.setattr("src.chat.sanitize_history", lambda h: h)
    monkeypatch.setattr("src.chat.vectordb.resolve_query_collection", lambda c, m, workspace_id="default": c)
    monkeypatch.setattr("src.chat.get_settings", lambda: settings)

    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: True)
    monkeypatch.setattr("src.structured_store.query_structured", lambda *a, **k: [
        {"doc_id": "rol.pdf", "row_index": 10, "texto_canonico": "codigo: 10101039; autorizacao: sim", "page_number": 3}
    ])
    monkeypatch.setattr("src.chat._vector_retrieve", lambda *a, **k: [])

    out = chat._retrieve_with_routing(
        question="codigo 10101039 precisa autorizacao?",
        collection="documentos",
        physical_collection="documentos",
        settings=settings,
        top_k=5,
    )
    assert out
    assert out[0]["metadata"]["chunk_type"] == "tabular_structured"


def test_retrieve_with_routing_fallback_to_vector_when_no_structured_hits(monkeypatch):
    settings = _settings(query_routing_enabled=True, structured_store_enabled=True)
    monkeypatch.setattr("src.structured_store.has_structured_data", lambda c: True)
    monkeypatch.setattr("src.structured_store.query_structured", lambda *a, **k: [])
    monkeypatch.setattr(
        "src.chat._vector_retrieve",
        lambda **kwargs: [{"id": "v1", "text": "resultado vetorial", "metadata": {"doc_id": "doc"}, "score": 0.7}],
    )
    out = chat._retrieve_with_routing(
        question="codigo 10101039 precisa autorizacao?",
        collection="documentos",
        physical_collection="documentos",
        settings=settings,
        top_k=5,
    )
    assert out[0]["id"] == "v1"


# ── Deduplication tests ────────────────────────────────────────────────────


def test_deduplicate_hits_keeps_highest_score():
    hits = [
        {"id": "c1", "text": "a", "score": 0.5},
        {"id": "c1", "text": "a", "score": 0.8},
        {"id": "c2", "text": "b", "score": 0.3},
    ]
    deduped = chat._deduplicate_hits(hits)
    assert len(deduped) == 2
    c1 = [h for h in deduped if h["id"] == "c1"][0]
    assert c1["score"] == 0.8


# ── Query expansion tests ───────────────────────────────────────────────────


def test_expand_query_llm_returns_original_plus_expansions(monkeypatch):
    monkeypatch.setattr(
        "src.chat.llm.chat",
        lambda messages, system="": "Qual o prazo da rescisao trabalhista?\nEm quanto tempo deve ser pago o valor rescisorio?",
    )
    settings = _settings(query_expansion_max_reformulations=2)
    result = chat._expand_query_llm("Prazo de rescisao?", settings)
    assert result[0] == "Prazo de rescisao?"
    assert len(result) == 3


def test_expand_query_llm_fallback_on_error(monkeypatch):
    monkeypatch.setattr("src.chat.llm.chat", lambda messages, system="": (_ for _ in ()).throw(RuntimeError("API error")))
    settings = _settings(query_expansion_max_reformulations=2)
    result = chat._expand_query_llm("teste", settings)
    assert result == ["teste"]


# ── HyDE tests ──────────────────────────────────────────────────────────────


def test_generate_hypothetical_doc_returns_text(monkeypatch):
    monkeypatch.setattr(
        "src.chat.llm.chat",
        lambda messages, system="": "O prazo para pagamento de rescisao e regulado pela CLT.",
    )
    settings = _settings()
    result = chat._generate_hypothetical_doc("Prazo de rescisao?", settings)
    assert "rescisao" in result.lower()


def test_generate_hypothetical_doc_fallback_on_error(monkeypatch):
    monkeypatch.setattr("src.chat.llm.chat", lambda messages, system="": (_ for _ in ()).throw(RuntimeError("fail")))
    settings = _settings()
    result = chat._generate_hypothetical_doc("Prazo?", settings)
    assert result == "Prazo?"


def test_vector_retrieve_structural_query_expands_reranker_top_k(monkeypatch):
    settings = _settings(
        reranker_enabled=True,
        reranker_top_k=12,
    )
    monkeypatch.setattr("src.chat.llm.embed", lambda texts, model_name=None: [[0.1, 0.2]])
    monkeypatch.setattr(
        "src.chat.vectordb.query",
        lambda *args, **kwargs: [
            {
                "id": f"chunk-{i}",
                "text": f"Trecho {i}",
                "metadata": {"doc_id": "doc", "section": "CAPITULO I" if i == 20 else ""},
                "distance": 0.2,
            }
            for i in range(54)
        ],
    )

    captured = {"top_k": None}

    def fake_rerank(query, candidates, model_name, top_k):
        captured["top_k"] = top_k
        return candidates[:top_k]

    monkeypatch.setattr("src.reranker.rerank", fake_rerank)

    out = chat._vector_retrieve(
        question="Resuma o capitulo 1",
        physical_collection="c",
        top_k=18,
        settings=settings,
        where=None,
        model_name="m",
    )

    assert captured["top_k"] == 54
    assert len(out) <= 18
