from src.prompts import build_rag_messages, format_context, get_rag_system, RAG_SYSTEM


def test_rag_messages_contains_context_and_question():
    msgs = build_rag_messages(context="Artigo 477 da CLT...", question="Qual o prazo?")
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "Artigo 477 da CLT" in msgs[0]["content"]
    assert "Qual o prazo?" in msgs[0]["content"]


def test_rag_system_is_in_portuguese():
    assert "portugues" in RAG_SYSTEM.lower() or "documentos" in RAG_SYSTEM.lower()


def test_rag_messages_format():
    msgs = build_rag_messages(context="ctx", question="q")
    assert isinstance(msgs, list)
    assert all("role" in m and "content" in m for m in msgs)


def test_rag_system_has_citation_instructions():
    assert "[N]" in RAG_SYSTEM or "[1]" in RAG_SYSTEM


def test_get_rag_system_applies_domain_profile():
    system = get_rag_system("legal")
    assert "legal" in system.lower()
    assert "claus" in system.lower() or "jurid" in system.lower()


def test_format_context_includes_doc_id():
    items = [
        {"id": "c1", "text": "Trecho sobre ferias.", "metadata": {"doc_id": "politica.pdf"}},
        {"id": "c2", "text": "Trecho sobre rescisao.", "metadata": {"doc_id": "clt.pdf"}},
    ]
    ctx = format_context(items)
    assert "politica.pdf" in ctx
    assert "clt.pdf" in ctx
    assert "[1]" in ctx
    assert "[2]" in ctx


def test_format_context_handles_missing_metadata():
    items = [{"id": "c1", "text": "Trecho sem metadata."}]
    ctx = format_context(items)
    assert "desconhecido" in ctx
    assert "[1]" in ctx


def test_format_context_numbering():
    items = [
        {"id": f"c{i}", "text": f"Chunk {i}", "metadata": {"doc_id": f"doc{i}.pdf"}}
        for i in range(5)
    ]
    ctx = format_context(items)
    for i in range(1, 6):
        assert f"[{i}]" in ctx


def test_format_context_includes_page_number():
    items = [
        {"id": "c1", "text": "Trecho com pagina.", "metadata": {"doc_id": "doc.pdf", "page_number": 3}},
    ]
    ctx = format_context(items)
    assert "p. 3" in ctx
    assert "doc.pdf" in ctx


def test_format_context_omits_page_when_not_available():
    items = [
        {"id": "c1", "text": "Trecho sem pagina.", "metadata": {"doc_id": "doc.txt"}},
    ]
    ctx = format_context(items)
    assert "p." not in ctx
    assert "doc.txt" in ctx


def test_format_context_groups_parent_child():
    items = [
        {
            "id": "parent-1",
            "text": "Art. 41° Texto completo do artigo.",
            "metadata": {"doc_id": "estatuto.pdf", "chunk_type": "parent", "parent_key": "doc::Art. 41", "artigo": "Art. 41", "caminho_hierarquico": "Cap V > Art. 41"},
        },
        {
            "id": "child-1",
            "text": "§ 1° Parágrafo do artigo 41.",
            "metadata": {"doc_id": "estatuto.pdf", "chunk_type": "child", "parent_key": "doc::Art. 41", "artigo": "Art. 41"},
        },
        {
            "id": "other",
            "text": "Outro trecho.",
            "metadata": {"doc_id": "outro.pdf"},
        },
    ]
    ctx = format_context(items)
    # Parent and child should be grouped into a single [N] block
    assert ctx.count("[1]") == 1
    assert ctx.count("[2]") == 1
    # Should NOT have a [3] since parent+child are grouped
    assert "[3]" not in ctx
    # The grouped block should contain both texts
    assert "Art. 41" in ctx
    assert "§ 1°" in ctx
    assert "Cap V > Art. 41" in ctx


def test_format_context_includes_artigo_in_label():
    items = [
        {
            "id": "c1",
            "text": "Texto do artigo.",
            "metadata": {"doc_id": "doc.pdf", "artigo": "Art. 5"},
        },
    ]
    ctx = format_context(items)
    assert "Art. 5" in ctx


def test_format_context_includes_paragraph_and_inciso_in_label():
    items = [
        {
            "id": "c1",
            "text": "II - comparecer as assembleias.",
            "metadata": {
                "doc_id": "estatuto.pdf",
                "artigo": "Art. 7",
                "paragrafo": "caput",
                "inciso": "II",
            },
        },
    ]
    ctx = format_context(items)
    assert "Art. 7" in ctx
    assert "caput" in ctx
    assert "inciso II" in ctx
    assert "Trecho:" in ctx
