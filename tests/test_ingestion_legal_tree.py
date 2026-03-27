"""Tests for legal tree integration in ingestion pipeline (config flags, cache)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.legal_tree import LegalNode, LegalTree, build_legal_tree


def _make_mock_tree():
    """Build a real (small) LegalTree for testing."""
    text = """\
CAPÍTULO I - DAS OBRIGAÇÕES

Art. 1° - O empregado deve cumprir horário.
Art. 2° - O empregado deve usar uniforme.
"""
    return build_legal_tree(text, doc_id="doc1", doc_name="Teste")


def _mock_metrics():
    m = MagicMock()
    m.time_block.return_value.__enter__ = MagicMock()
    m.time_block.return_value.__exit__ = MagicMock(return_value=False)
    return m


class TestBuildAndIndexLegalTreeFlags:
    """Test that _build_and_index_legal_tree respects config flags."""

    @patch("src.summaries.generate_tree_summaries")
    @patch("src.controlplane.delete_document_nodes")
    @patch("src.controlplane.upsert_document_node")
    @patch("src.controlplane.list_document_summaries")
    @patch("src.controlplane.delete_document_summaries")
    @patch("src.controlplane.upsert_document_summary")
    def test_summaries_disabled_skips_generation(
        self, mock_upsert_sum, mock_del_sum, mock_list_sum,
        mock_upsert_node, mock_del_nodes, mock_gen_summaries,
    ):
        """When legal_summaries_enabled=False, summaries are not generated."""
        from src.ingestion import _build_and_index_legal_tree

        settings = MagicMock()
        settings.legal_summaries_enabled = False
        settings.legal_summaries_cache = True

        mock_llm = MagicMock()
        mock_llm.embed.return_value = [[0.1] * 768]

        _build_and_index_legal_tree(
            clean_text="CAPÍTULO I - DAS OBRIGAÇÕES\n\nArt. 1° - Teste.",
            doc_id="doc1",
            collection="col1",
            physical_collection="col1",
            workspace_id="ws1",
            model_name="model",
            llm_mod=mock_llm,
            vectordb_mod=MagicMock(),
            metrics=_mock_metrics(),
            settings=settings,
        )

        # Summaries should NOT be generated
        mock_gen_summaries.assert_not_called()
        # But nodes should still be persisted
        mock_upsert_node.assert_called()

    @patch("src.summaries.generate_tree_summaries")
    @patch("src.controlplane.delete_document_nodes")
    @patch("src.controlplane.upsert_document_node")
    @patch("src.controlplane.list_document_summaries")
    @patch("src.controlplane.delete_document_summaries")
    @patch("src.controlplane.upsert_document_summary")
    def test_summaries_cache_hit_skips_generation(
        self, mock_upsert_sum, mock_del_sum, mock_list_sum,
        mock_upsert_node, mock_del_nodes, mock_gen_summaries,
    ):
        """When cache is enabled and summaries exist, skip LLM calls."""
        from src.ingestion import _build_and_index_legal_tree

        # Cache returns existing summaries
        mock_list_sum.return_value = [{"node_id": "n1", "label": "CAP I"}]

        settings = MagicMock()
        settings.legal_summaries_enabled = True
        settings.legal_summaries_cache = True

        mock_llm = MagicMock()
        mock_llm.embed.return_value = [[0.1] * 768]

        _build_and_index_legal_tree(
            clean_text="CAPÍTULO I - DAS OBRIGAÇÕES\n\nArt. 1° - Teste.",
            doc_id="doc1",
            collection="col1",
            physical_collection="col1",
            workspace_id="ws1",
            model_name="model",
            llm_mod=mock_llm,
            vectordb_mod=MagicMock(),
            metrics=_mock_metrics(),
            settings=settings,
        )

        # Should check cache
        mock_list_sum.assert_called_once_with("ws1", "col1", "doc1")
        # Should NOT generate summaries
        mock_gen_summaries.assert_not_called()

    @patch("src.summaries.generate_tree_summaries")
    @patch("src.controlplane.delete_document_nodes")
    @patch("src.controlplane.upsert_document_node")
    @patch("src.controlplane.list_document_summaries")
    @patch("src.controlplane.delete_document_summaries")
    @patch("src.controlplane.upsert_document_summary")
    def test_summaries_cache_miss_generates(
        self, mock_upsert_sum, mock_del_sum, mock_list_sum,
        mock_upsert_node, mock_del_nodes, mock_gen_summaries,
    ):
        """When cache is enabled but empty, generate summaries normally."""
        from src.ingestion import _build_and_index_legal_tree

        mock_list_sum.return_value = []

        mock_summary = MagicMock()
        mock_summary.node_id = "n1"
        mock_summary.node_type = "capitulo"
        mock_summary.label = "CAP I"
        mock_summary.path = "CAP I"
        mock_summary.resumo_executivo = "Resumo"
        mock_summary.resumo_juridico = "Jurídico"
        mock_summary.pontos_chave = ["a"]
        mock_summary.artigos_cobertos = ["1"]
        mock_summary.obrigacoes = []
        mock_summary.restricoes = []
        mock_summary.definicoes = []
        mock_summary.text_length = 100
        mock_summary.source_hash = "abc"
        mock_summary.source_text_length = 100
        mock_summary.status = "generated"
        mock_summary.validation_errors = []
        mock_summary.generation_meta = {}
        mock_gen_summaries.return_value = [mock_summary]

        settings = MagicMock()
        settings.legal_summaries_enabled = True
        settings.legal_summaries_cache = True

        mock_llm = MagicMock()
        mock_llm.embed.return_value = [[0.1] * 768]

        _build_and_index_legal_tree(
            clean_text="CAPÍTULO I - DAS OBRIGAÇÕES\n\nArt. 1° - Teste.",
            doc_id="doc1",
            collection="col1",
            physical_collection="col1",
            workspace_id="ws1",
            model_name="model",
            llm_mod=mock_llm,
            vectordb_mod=MagicMock(),
            metrics=_mock_metrics(),
            settings=settings,
        )

        # Should generate summaries since cache was empty
        mock_gen_summaries.assert_called_once()
        # Should persist them
        mock_upsert_sum.assert_called_once()

    @patch("src.summaries.generate_tree_summaries")
    @patch("src.controlplane.delete_document_nodes")
    @patch("src.controlplane.upsert_document_node")
    @patch("src.controlplane.list_document_summaries")
    @patch("src.controlplane.delete_document_summaries")
    @patch("src.controlplane.upsert_document_summary")
    def test_no_settings_defaults_to_enabled(
        self, mock_upsert_sum, mock_del_sum, mock_list_sum,
        mock_upsert_node, mock_del_nodes, mock_gen_summaries,
    ):
        """When settings=None, summaries are generated (backward compat)."""
        from src.ingestion import _build_and_index_legal_tree

        mock_list_sum.return_value = []

        mock_summary = MagicMock()
        mock_summary.node_id = "n1"
        mock_summary.node_type = "capitulo"
        mock_summary.label = "CAP I"
        mock_summary.path = "CAP I"
        mock_summary.resumo_executivo = "Resumo"
        mock_summary.resumo_juridico = ""
        mock_summary.pontos_chave = []
        mock_summary.artigos_cobertos = []
        mock_summary.obrigacoes = []
        mock_summary.restricoes = []
        mock_summary.definicoes = []
        mock_summary.text_length = 50
        mock_summary.source_hash = "def"
        mock_summary.source_text_length = 50
        mock_summary.status = "generated"
        mock_summary.validation_errors = []
        mock_summary.generation_meta = {}
        mock_gen_summaries.return_value = [mock_summary]

        mock_llm = MagicMock()
        mock_llm.embed.return_value = [[0.1] * 768]

        _build_and_index_legal_tree(
            clean_text="CAPÍTULO I - DAS OBRIGAÇÕES\n\nArt. 1° - Teste.",
            doc_id="doc1",
            collection="col1",
            physical_collection="col1",
            workspace_id="ws1",
            model_name="model",
            llm_mod=mock_llm,
            vectordb_mod=MagicMock(),
            metrics=_mock_metrics(),
            settings=None,
        )

        # Should still generate summaries (default behavior)
        mock_gen_summaries.assert_called_once()
