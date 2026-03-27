"""Tests for intent classification in src/chat.py."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.chat import (
    QueryIntent,
    _classify_intent_regex,
    _cosine_similarity,
)


# ── Regex-based classifier ─────────────────────────────────────────────────

class TestClassifyIntentRegex:
    def test_summary_structural(self):
        assert _classify_intent_regex("Resuma o capítulo II") == QueryIntent.SUMMARY_STRUCTURAL

    def test_summary_with_sintetize(self):
        assert _classify_intent_regex("Sintetize a seção III") == QueryIntent.SUMMARY_STRUCTURAL

    def test_summary_visao_geral(self):
        assert _classify_intent_regex("Visão geral do título I") == QueryIntent.SUMMARY_STRUCTURAL

    def test_question_structural(self):
        assert _classify_intent_regex("O que diz o capítulo 5 sobre obrigações?") == QueryIntent.QUESTION_STRUCTURAL

    def test_question_factual(self):
        assert _classify_intent_regex("Qual o prazo para pagamento?") == QueryIntent.QUESTION_FACTUAL

    def test_comparison(self):
        assert _classify_intent_regex("Compare o capítulo 1 e o capítulo 2") == QueryIntent.COMPARISON

    def test_locate_excerpt_short(self):
        # Short query with just structural ref → LOCATE_EXCERPT
        result = _classify_intent_regex("Capítulo II")
        assert result in (QueryIntent.LOCATE_EXCERPT, QueryIntent.QUESTION_STRUCTURAL)

    def test_factual_no_structural_ref(self):
        assert _classify_intent_regex("É proibido remunerar diretores?") == QueryIntent.QUESTION_FACTUAL

    def test_empty_question(self):
        assert _classify_intent_regex("") == QueryIntent.QUESTION_FACTUAL


# ── classify_query_intent dispatch ──────────────────────────────────────────

class TestClassifyQueryIntentDispatch:
    @patch("src.chat.get_settings")
    def test_dispatch_to_regex(self, mock_settings):
        from src.chat import classify_query_intent
        settings = MagicMock()
        settings.intent_classifier = "regex"
        mock_settings.return_value = settings
        result = classify_query_intent("Qual o prazo?")
        assert result == QueryIntent.QUESTION_FACTUAL

    @patch("src.chat._classify_intent_embeddings")
    @patch("src.chat.get_settings")
    def test_dispatch_to_embeddings(self, mock_settings, mock_emb_classify):
        from src.chat import classify_query_intent
        settings = MagicMock()
        settings.intent_classifier = "embeddings"
        mock_settings.return_value = settings
        mock_emb_classify.return_value = QueryIntent.SUMMARY_STRUCTURAL
        result = classify_query_intent("Resuma o capítulo II")
        mock_emb_classify.assert_called_once()
        assert result == QueryIntent.SUMMARY_STRUCTURAL


# ── Cosine similarity ──────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── Embeddings-based classifier (mocked LLM) ──────────────────────────────

class TestClassifyIntentEmbeddings:
    @patch("src.chat._intent_exemplar_embeddings", None)
    @patch("src.chat.llm")
    def test_classifies_summary_structural(self, mock_llm):
        """With mocked embeddings, the classifier should pick the intent with
        highest similarity. We simulate by returning a vector close to
        summary_structural exemplars."""
        from src.chat import _classify_intent_embeddings, _INTENT_EXEMPLARS
        import src.chat as chat_mod

        # Reset cache
        chat_mod._intent_exemplar_embeddings = None

        # Unique vector per intent, query vector matches summary_structural
        intent_vectors = {
            QueryIntent.SUMMARY_STRUCTURAL: [1.0, 0.0, 0.0],
            QueryIntent.QUESTION_STRUCTURAL: [0.0, 1.0, 0.0],
            QueryIntent.QUESTION_FACTUAL: [0.0, 0.0, 1.0],
            QueryIntent.LOCATE_EXCERPT: [0.5, 0.5, 0.0],
            QueryIntent.COMPARISON: [0.0, 0.5, 0.5],
        }

        def mock_embed(texts, **kwargs):
            # Return vectors based on intent exemplar membership
            results = []
            for text in texts:
                found = False
                for intent, exemplars in _INTENT_EXEMPLARS.items():
                    if text in exemplars:
                        results.append(intent_vectors[intent])
                        found = True
                        break
                if not found:
                    # Query text → close to summary_structural
                    results.append([0.95, 0.1, 0.05])
            return results

        mock_llm.embed = mock_embed

        result = _classify_intent_embeddings("Resuma o capítulo II")
        assert result == QueryIntent.SUMMARY_STRUCTURAL

        # Clean up
        chat_mod._intent_exemplar_embeddings = None
