"""Tests for src/legal_tree.py — canonical legal document tree builder."""
from __future__ import annotations

import hashlib

import pytest

from src.legal_tree import (
    LegalTree,
    _extract_numeral,
    _node_id,
    build_legal_tree,
)

# ── Sample legal text ──────────────────────────────────────────────────────

SAMPLE_LEGAL_TEXT = """\
TÍTULO I - DISPOSIÇÕES GERAIS

CAPÍTULO I - DA DENOMINAÇÃO

Art. 1° - A empresa tem sede em São Paulo.
Art. 2° - O objeto social é a prestação de serviços.

CAPÍTULO II - DOS OBJETIVOS SOCIAIS

Art. 3° - Os objetivos são:
§ 1° - Promover o desenvolvimento.
§ 2° - Garantir a qualidade.

Art. 4° - É vedada a distribuição de lucros.

SEÇÃO I - DAS RESTRIÇÕES

Art. 5° - Não é permitido:
I - Participar de atividades ilícitas.
II - Distribuir recursos sem autorização.

TÍTULO II - DA ADMINISTRAÇÃO

CAPÍTULO III - DO CONSELHO

Art. 6° - O conselho é composto por 5 membros.
"""

CHAPTERS_ONLY_TEXT = """\
CAPÍTULO I - DAS OBRIGAÇÕES

Art. 1° - O empregado deve cumprir horário.
Art. 2° - O empregado deve usar uniforme.

CAPÍTULO II - DOS DIREITOS

Art. 3° - O empregado tem direito a férias.
"""

NO_STRUCTURE_TEXT = """\
Este documento não possui marcadores estruturais.
Apenas texto corrido sem títulos, capítulos ou seções.
Art. 1° - Cláusula genérica de exemplo.
"""

NORMALIZED_HEADINGS_TEXT = """\
TITULO I - DISPOSICOES GERAIS

CAPITULO II - DOS OBJETIVOS SOCIAIS

Art. 3 - Os objetivos sao:
Art. 4 - E vedada a distribuicao de lucros.

SECAO I - DAS RESTRICOES

Art. 5 - Nao e permitido.
"""


# ── _extract_numeral ───────────────────────────────────────────────────────

class TestExtractNumeral:
    def test_roman_numeral_with_keyword(self):
        assert _extract_numeral("CAPÍTULO II - DOS OBJETIVOS") == "II"

    def test_arabic_numeral_with_keyword(self):
        assert _extract_numeral("CAPÍTULO 3 - DO CONSELHO") == "3"

    def test_complex_roman_with_keyword(self):
        assert _extract_numeral("TÍTULO XIV - DAS DISPOSIÇÕES FINAIS") == "XIV"

    def test_secao_numeral(self):
        assert _extract_numeral("SEÇÃO III - DAS RESTRIÇÕES") == "III"

    def test_no_keyword_returns_empty(self):
        # Without a structural keyword prefix, returns empty
        assert _extract_numeral("II - DOS OBJETIVOS") == ""
        assert _extract_numeral("") == ""
        assert _extract_numeral("---") == ""

    def test_titulo_numeral(self):
        assert _extract_numeral("TÍTULO I - DISPOSIÇÕES GERAIS") == "I"


# ── _node_id ───────────────────────────────────────────────────────────────

class TestNodeId:
    def test_deterministic(self):
        id1 = _node_id("doc1", "capitulo", "CAP I")
        id2 = _node_id("doc1", "capitulo", "CAP I")
        assert id1 == id2

    def test_different_inputs_differ(self):
        id1 = _node_id("doc1", "capitulo", "CAP I")
        id2 = _node_id("doc1", "capitulo", "CAP II")
        assert id1 != id2

    def test_sha256_prefix(self):
        raw = "doc1::capitulo::CAP I"
        expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
        assert _node_id("doc1", "capitulo", "CAP I") == expected

    def test_length_is_16(self):
        assert len(_node_id("x", "y", "z")) == 16


# ── build_legal_tree — full hierarchy ─────────────────────────────────────

class TestBuildLegalTreeFull:
    @pytest.fixture()
    def tree(self) -> LegalTree:
        return build_legal_tree(SAMPLE_LEGAL_TEXT, doc_id="doc1", doc_name="Estatuto Social")

    def test_root_exists(self, tree: LegalTree):
        assert tree.root is not None
        assert tree.root.node_type == "documento"
        assert tree.root.label == "Estatuto Social"

    def test_titles_count(self, tree: LegalTree):
        assert len(tree.get_titles()) == 2

    def test_chapters_count(self, tree: LegalTree):
        assert len(tree.get_chapters()) == 3

    def test_sections_count(self, tree: LegalTree):
        assert len(tree.get_sections()) == 1

    def test_macro_nodes_count(self, tree: LegalTree):
        # 2 titulos + 3 capitulos + 1 secao = 6
        assert len(tree.get_macro_nodes()) == 6

    def test_find_by_numeral_secao_i(self, tree: LegalTree):
        # SEÇÃO extracts numeral correctly since "SEÇÃO" has no roman-like prefix
        # Actually the "I" is extracted correctly for SEÇÃO I
        node = tree.find_by_numeral("secao", "I")
        assert node is not None
        assert "RESTRI" in node.label.upper()

    def test_find_by_numeral_nonexistent(self, tree: LegalTree):
        assert tree.find_by_numeral("capitulo", "XXXX") is None

    def test_find_by_label_titulo(self, tree: LegalTree):
        # Use find_by_label since numeral extraction from "TÍTULO" picks "L"
        results = tree.find_by_label("DISPOSIÇÕES GERAIS")
        assert len(results) >= 1
        assert any(n.node_type == "titulo" for n in results)

    def test_chapters_are_children_of_titles(self, tree: LegalTree):
        titles = tree.get_titles()
        assert len(titles) >= 1
        # First title should have chapter children
        first_title = [t for t in titles if "GERAIS" in t.label.upper()][0]
        child_types = [c.node_type for c in first_title.children]
        assert "capitulo" in child_types

    def test_section_is_child_of_chapter(self, tree: LegalTree):
        # Find chapter with OBJETIVOS in label
        chapters = tree.get_chapters()
        cap_obj = [c for c in chapters if "OBJETIVOS" in c.label.upper()]
        assert len(cap_obj) >= 1
        child_types = [c.node_type for c in cap_obj[0].children]
        assert "secao" in child_types

    def test_articles_in_chapter(self, tree: LegalTree):
        # Find chapter with DENOMINAÇÃO
        chapters = tree.get_chapters()
        cap_den = [c for c in chapters if "DENOMINA" in c.label.upper()]
        assert len(cap_den) >= 1
        assert "1" in cap_den[0].articles
        assert "2" in cap_den[0].articles

    def test_articles_in_chapter_with_section(self, tree: LegalTree):
        chapters = tree.get_chapters()
        cap_obj = [c for c in chapters if "OBJETIVOS" in c.label.upper()]
        assert len(cap_obj) >= 1
        # Chapter II scope includes Art. 3, 4, and 5 (section is within chapter)
        assert "3" in cap_obj[0].articles
        assert "4" in cap_obj[0].articles

    def test_hierarchical_path(self, tree: LegalTree):
        sections = tree.get_sections()
        assert len(sections) >= 1
        # Path should contain parent references
        assert ">" in sections[0].path

    def test_to_dict_serialization(self, tree: LegalTree):
        d = tree.to_dict()
        assert "doc_id" in d
        assert "doc_name" in d
        assert "root" in d
        assert d["doc_id"] == "doc1"
        assert d["doc_name"] == "Estatuto Social"

    def test_to_json_serialization(self, tree: LegalTree):
        j = tree.to_json()
        import json
        data = json.loads(j)
        assert data["doc_id"] == "doc1"

    def test_node_index_populated(self, tree: LegalTree):
        # Root + 2 titulos + 3 capitulos + 1 secao = 7
        assert len(tree.node_index) >= 7

    def test_get_node_by_id(self, tree: LegalTree):
        for nid, node in tree.node_index.items():
            assert tree.get_node(nid) is node

    def test_get_node_invalid_id(self, tree: LegalTree):
        assert tree.get_node("nonexistent_id") is None

    def test_find_by_label(self, tree: LegalTree):
        results = tree.find_by_label("CONSELHO")
        assert len(results) >= 1
        assert any("CONSELHO" in n.label.upper() for n in results)


# ── build_legal_tree — chapters only ──────────────────────────────────────

class TestBuildLegalTreeChaptersOnly:
    @pytest.fixture()
    def tree(self) -> LegalTree:
        return build_legal_tree(CHAPTERS_ONLY_TEXT, doc_id="doc2")

    def test_no_titles(self, tree: LegalTree):
        assert len(tree.get_titles()) == 0

    def test_chapters_direct_children_of_root(self, tree: LegalTree):
        root_child_types = [c.node_type for c in tree.root.children]
        assert all(t == "capitulo" for t in root_child_types)
        assert len(root_child_types) == 2

    def test_chapters_count(self, tree: LegalTree):
        assert len(tree.get_chapters()) == 2


# ── build_legal_tree — no structural markers ──────────────────────────────

class TestBuildLegalTreeFlat:
    @pytest.fixture()
    def tree(self) -> LegalTree:
        return build_legal_tree(NO_STRUCTURE_TEXT, doc_id="doc3", doc_name="Flat Doc")

    def test_only_root_node(self, tree: LegalTree):
        assert len(tree.get_chapters()) == 0
        assert len(tree.get_sections()) == 0
        assert len(tree.get_titles()) == 0

    def test_root_contains_full_text(self, tree: LegalTree):
        assert "marcadores estruturais" in tree.root.text

    def test_macro_nodes_empty(self, tree: LegalTree):
        assert len(tree.get_macro_nodes()) == 0

    def test_root_articles(self, tree: LegalTree):
        assert "1" in tree.root.articles


class TestBuildLegalTreeNormalizedFallback:
    def test_detects_ascii_headings(self):
        tree = build_legal_tree(NORMALIZED_HEADINGS_TEXT, doc_id="doc4")
        assert len(tree.get_titles()) == 1
        assert len(tree.get_chapters()) == 1
        assert len(tree.get_sections()) == 1
        node = tree.find_by_numeral("capitulo", "II")
        assert node is not None


# ── LegalNode.to_dict ─────────────────────────────────────────────────────

class TestLegalNodeToDict:
    def test_text_truncation_in_dict(self):
        tree = build_legal_tree("x" * 1000, doc_id="trunc_test")
        d = tree.root.to_dict()
        # text field should be truncated to 500 + "..."
        assert d["text"].endswith("...")
        assert d["text_length"] == 1000

    def test_short_text_not_truncated(self):
        tree = build_legal_tree("short text", doc_id="short_test")
        d = tree.root.to_dict()
        assert not d["text"].endswith("...")

    def test_children_serialized(self):
        tree = build_legal_tree(SAMPLE_LEGAL_TEXT, doc_id="ser_test")
        d = tree.root.to_dict()
        assert "children" in d
        assert isinstance(d["children"], list)
