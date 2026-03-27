"""Tests for src/summaries.py — pre-computed summaries for legal tree nodes."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.legal_tree import LegalNode, LegalTree, build_legal_tree
from src.summaries import (
    NodeSummary,
    _parse_llm_json,
    _truncate_text,
    build_summary_context,
    generate_node_summary,
    generate_tree_summaries,
)


# ── _parse_llm_json ───────────────────────────────────────────────────────

class TestParseLlmJson:
    def test_clean_json(self):
        raw = '{"resumo_executivo": "Teste", "pontos_chave": ["a", "b"]}'
        result = _parse_llm_json(raw)
        assert result["resumo_executivo"] == "Teste"
        assert result["pontos_chave"] == ["a", "b"]

    def test_markdown_code_block(self):
        raw = '```json\n{"resumo_executivo": "Teste"}\n```'
        result = _parse_llm_json(raw)
        assert result["resumo_executivo"] == "Teste"

    def test_markdown_code_block_no_lang(self):
        raw = '```\n{"resumo_executivo": "Bloco"}\n```'
        result = _parse_llm_json(raw)
        assert result["resumo_executivo"] == "Bloco"

    def test_invalid_json_returns_empty(self):
        result = _parse_llm_json("this is not json at all")
        assert result == {}

    def test_json_embedded_in_text(self):
        raw = 'Aqui vai o resultado:\n{"resumo_executivo": "Embutido"}\nFim.'
        result = _parse_llm_json(raw)
        assert result["resumo_executivo"] == "Embutido"

    def test_empty_string(self):
        assert _parse_llm_json("") == {}

    def test_nested_json(self):
        raw = '{"resumo_executivo": "X", "pontos_chave": ["a"], "obrigacoes": []}'
        result = _parse_llm_json(raw)
        assert result["resumo_executivo"] == "X"
        assert result["obrigacoes"] == []


# ── _truncate_text ─────────────────────────────────────────────────────────

class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Texto curto."
        assert _truncate_text(text, max_chars=100) == text

    def test_exact_limit_unchanged(self):
        text = "a" * 100
        assert _truncate_text(text, max_chars=100) == text

    def test_long_text_truncated(self):
        text = "a" * 200
        result = _truncate_text(text, max_chars=100)
        assert len(result) > 100  # Has the truncation note
        assert result.startswith("a" * 100)
        assert "truncado" in result

    def test_default_max_is_12000(self):
        text = "x" * 11999
        assert _truncate_text(text) == text
        text_long = "x" * 12001
        assert "truncado" in _truncate_text(text_long)


# ── NodeSummary serialization ──────────────────────────────────────────────

class TestNodeSummaryRoundtrip:
    def test_to_dict_and_from_dict(self):
        summary = NodeSummary(
            node_id="abc123",
            node_type="capitulo",
            label="CAPÍTULO I - DAS OBRIGAÇÕES",
            path="Título I > Capítulo I",
            resumo_executivo="Resumo executivo de teste.",
            resumo_juridico="Resumo jurídico de teste.",
            pontos_chave=["Ponto 1", "Ponto 2"],
            artigos_cobertos=["Art. 1", "Art. 2"],
            obrigacoes=["Obrigação A"],
            restricoes=["Restrição X"],
            definicoes=["Termo: Definição"],
            text_length=500,
        )
        d = summary.to_dict()
        restored = NodeSummary.from_dict(d)

        assert restored.node_id == summary.node_id
        assert restored.node_type == summary.node_type
        assert restored.label == summary.label
        assert restored.path == summary.path
        assert restored.resumo_executivo == summary.resumo_executivo
        assert restored.resumo_juridico == summary.resumo_juridico
        assert restored.pontos_chave == summary.pontos_chave
        assert restored.artigos_cobertos == summary.artigos_cobertos
        assert restored.obrigacoes == summary.obrigacoes
        assert restored.restricoes == summary.restricoes
        assert restored.definicoes == summary.definicoes
        assert restored.text_length == summary.text_length

    def test_from_dict_with_missing_keys(self):
        d = {"node_id": "x", "label": "Test"}
        s = NodeSummary.from_dict(d)
        assert s.node_id == "x"
        assert s.label == "Test"
        assert s.node_type == ""
        assert s.pontos_chave == []

    def test_to_dict_contains_all_keys(self):
        s = NodeSummary(node_id="a", node_type="b", label="c", path="d")
        d = s.to_dict()
        expected_keys = {
            "node_id", "node_type", "label", "path",
            "resumo_executivo", "resumo_juridico",
            "pontos_chave", "artigos_cobertos",
            "obrigacoes", "restricoes", "definicoes", "text_length",
        }
        assert set(d.keys()) == expected_keys


# ── build_summary_context ─────────────────────────────────────────────────

class TestBuildSummaryContext:
    def test_basic_format(self):
        summary = NodeSummary(
            node_id="id1",
            node_type="capitulo",
            label="CAPÍTULO I - DAS OBRIGAÇÕES",
            path="Título I > Capítulo I",
            resumo_executivo="Trata das obrigações do empregado.",
            resumo_juridico="Dispõe sobre deveres trabalhistas.",
            pontos_chave=["Ponto A", "Ponto B"],
            artigos_cobertos=["Art. 1", "Art. 2"],
            obrigacoes=["Cumprir horário"],
            restricoes=["Proibido fumar"],
            definicoes=["Empregado: pessoa física"],
        )
        ctx = build_summary_context(summary)
        assert "=== CAPÍTULO I - DAS OBRIGAÇÕES ===" in ctx
        assert "Caminho: Título I > Capítulo I" in ctx
        assert "Resumo executivo:" in ctx
        assert "Trata das obrigações do empregado." in ctx
        assert "Resumo juridico:" in ctx
        assert "Pontos-chave:" in ctx
        assert "- Ponto A" in ctx
        assert "Artigos cobertos: Art. 1, Art. 2" in ctx
        assert "Obrigacoes:" in ctx
        assert "- Cumprir horário" in ctx
        assert "Restricoes/vedacoes:" in ctx
        assert "- Proibido fumar" in ctx
        assert "Definicoes:" in ctx
        assert "- Empregado: pessoa física" in ctx

    def test_minimal_summary(self):
        summary = NodeSummary(
            node_id="id2",
            node_type="secao",
            label="SEÇÃO I",
            path="Seção I",
        )
        ctx = build_summary_context(summary)
        assert "=== SEÇÃO I ===" in ctx
        assert "Caminho: Seção I" in ctx
        # No optional sections should appear
        assert "Resumo executivo:" not in ctx
        assert "Pontos-chave:" not in ctx


# ── generate_node_summary (mocked LLM) ────────────────────────────────────

class TestGenerateNodeSummary:
    def _make_node(self) -> LegalNode:
        return LegalNode(
            id="test_node_id",
            node_type="capitulo",
            label="CAPÍTULO I - DAS OBRIGAÇÕES",
            numeral="I",
            text="Art. 1° - O empregado deve cumprir horário.\nArt. 2° - O empregado deve usar uniforme.",
            path="Título I > Capítulo I",
            articles=["1", "2"],
        )

    def test_with_mocked_llm(self):
        mock_response = json.dumps({
            "resumo_executivo": "Capítulo trata das obrigações do empregado.",
            "resumo_juridico": "Estabelece deveres trabalhistas conforme CLT.",
            "pontos_chave": ["Cumprimento de horário", "Uso de uniforme"],
            "artigos_cobertos": ["Art. 1", "Art. 2"],
            "obrigacoes": ["Cumprir horário", "Usar uniforme"],
            "restricoes": [],
            "definicoes": [],
        })

        node = self._make_node()
        with patch("src.llm.chat", return_value=mock_response):
            summary = generate_node_summary(node)

        assert summary.node_id == "test_node_id"
        assert summary.node_type == "capitulo"
        assert summary.label == "CAPÍTULO I - DAS OBRIGAÇÕES"
        assert summary.resumo_executivo == "Capítulo trata das obrigações do empregado."
        assert summary.pontos_chave == ["Cumprimento de horário", "Uso de uniforme"]
        assert summary.artigos_cobertos == ["Art. 1", "Art. 2"]
        assert summary.text_length == len(node.text)

    def test_llm_failure_returns_empty_summary(self):
        node = self._make_node()
        with patch("src.llm.chat", side_effect=RuntimeError("LLM unreachable")):
            summary = generate_node_summary(node)

        assert summary.node_id == "test_node_id"
        assert summary.resumo_executivo == ""
        assert summary.pontos_chave == []

    def test_llm_returns_invalid_json(self):
        node = self._make_node()
        with patch("src.llm.chat", return_value="not valid json at all"):
            summary = generate_node_summary(node)

        assert summary.resumo_executivo == ""


# ── generate_tree_summaries (mocked LLM) ──────────────────────────────────

TREE_TEXT = """\
CAPÍTULO I - DAS OBRIGAÇÕES

Art. 1° - O empregado deve cumprir horário.
Art. 2° - O empregado deve usar uniforme.

CAPÍTULO II - DOS DIREITOS

Art. 3° - O empregado tem direito a férias.
Art. 4° - O empregado tem direito a 13o salário.
"""


class TestGenerateTreeSummaries:
    def test_generates_summaries_for_macro_nodes(self):
        tree = build_legal_tree(TREE_TEXT, doc_id="tree_test")
        mock_response = json.dumps({
            "resumo_executivo": "Resumo teste.",
            "resumo_juridico": "Jurídico teste.",
            "pontos_chave": ["Ponto"],
            "artigos_cobertos": [],
            "obrigacoes": [],
            "restricoes": [],
            "definicoes": [],
        })

        with patch("src.llm.chat", return_value=mock_response):
            summaries = generate_tree_summaries(tree)

        # 2 chapters
        assert len(summaries) == 2
        assert all(s.resumo_executivo == "Resumo teste." for s in summaries)

    def test_skips_short_nodes(self):
        # Build a tree where a node has very short text
        tree = build_legal_tree(TREE_TEXT, doc_id="short_test")
        # Artificially shorten a node
        for node in tree.node_index.values():
            if node.node_type == "capitulo":
                node.text = "short"
                break

        mock_response = json.dumps({"resumo_executivo": "OK"})
        with patch("src.llm.chat", return_value=mock_response):
            summaries = generate_tree_summaries(tree)

        # One chapter was too short, only the other should get a summary
        assert len(summaries) == 1
