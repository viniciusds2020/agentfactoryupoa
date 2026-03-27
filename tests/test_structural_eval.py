"""Tests for structural evaluation functions in src/evaluation.py."""
from __future__ import annotations

import pytest

from src.evaluation import (
    article_coverage,
    citation_faithfulness,
    evaluate_structural_summary,
    section_boundary_precision,
    structural_hit_at_1,
)


# ── structural_hit_at_1 ───────────────────────────────────────────────────

class TestStructuralHitAt1:
    def test_exact_match(self):
        assert structural_hit_at_1("capitulo", "II", "capitulo", "II") == 1.0

    def test_mismatch_numeral(self):
        assert structural_hit_at_1("capitulo", "II", "capitulo", "III") == 0.0

    def test_mismatch_type(self):
        assert structural_hit_at_1("secao", "II", "capitulo", "II") == 0.0

    def test_case_insensitive_type(self):
        assert structural_hit_at_1("Capitulo", "II", "capitulo", "II") == 1.0

    def test_case_insensitive_numeral(self):
        assert structural_hit_at_1("capitulo", "ii", "capitulo", "II") == 1.0

    def test_whitespace_in_numeral(self):
        assert structural_hit_at_1("capitulo", " II ", "capitulo", "II") == 1.0

    def test_both_wrong(self):
        assert structural_hit_at_1("titulo", "X", "capitulo", "II") == 0.0

    def test_roman_arabic_equivalence(self):
        assert structural_hit_at_1("capitulo", "II", "capitulo", "2") == 1.0
        assert structural_hit_at_1("capitulo", "2", "capitulo", "II") == 1.0


# ── article_coverage ──────────────────────────────────────────────────────

class TestArticleCoverage:
    def test_full_coverage(self):
        assert article_coverage(
            ["Art. 1", "Art. 2", "Art. 3"],
            ["Art. 1", "Art. 2", "Art. 3"],
        ) == 1.0

    def test_partial_coverage(self):
        result = article_coverage(
            ["Art. 1", "Art. 3"],
            ["Art. 1", "Art. 2", "Art. 3"],
        )
        assert abs(result - 2 / 3) < 1e-9

    def test_no_coverage(self):
        assert article_coverage(
            ["Art. 10"],
            ["Art. 1", "Art. 2"],
        ) == 0.0

    def test_empty_expected_returns_1(self):
        assert article_coverage(["Art. 1"], []) == 1.0

    def test_empty_summary_articles(self):
        assert article_coverage([], ["Art. 1"]) == 0.0

    def test_both_empty(self):
        assert article_coverage([], []) == 1.0

    def test_case_insensitive(self):
        assert article_coverage(["art. 1"], ["Art. 1"]) == 1.0

    def test_whitespace_handling(self):
        assert article_coverage([" Art. 1 "], ["Art. 1"]) == 1.0


# ── citation_faithfulness ─────────────────────────────────────────────────

class TestCitationFaithfulness:
    def test_label_present(self):
        answer = "Conforme o CAPÍTULO II, as vedações incluem [1] distribuição de lucros."
        assert citation_faithfulness(answer, "CAPÍTULO II") == 1.0

    def test_label_absent(self):
        answer = "As vedações incluem [1] distribuição de lucros."
        result = citation_faithfulness(answer, "CAPÍTULO XV - DAS DISPOSIÇÕES")
        assert result == 0.0

    def test_no_citations_returns_zero(self):
        answer = "Texto sem citações."
        assert citation_faithfulness(answer, "CAPÍTULO I") == 0.0

    def test_partial_label_match(self):
        # The heuristic checks parts of the label (words > 3 chars)
        answer = "O conselho [1] delibera sobre questões administrativas."
        result = citation_faithfulness(answer, "CAPÍTULO III - DO CONSELHO")
        # "CONSELHO" (len > 3) should match
        assert result == 1.0

    def test_case_insensitive(self):
        answer = "No capítulo ii [1] temos as obrigações."
        assert citation_faithfulness(answer, "CAPÍTULO II") == 1.0


# ── section_boundary_precision ────────────────────────────────────────────

class TestSectionBoundaryPrecision:
    def test_no_contamination(self):
        answer = "O Capítulo I trata das obrigações."
        result = section_boundary_precision(
            answer,
            "CAPÍTULO I",
            ["CAPÍTULO II", "CAPÍTULO III"],
        )
        assert result == 1.0

    def test_full_contamination(self):
        answer = "Conforme Capítulo II e Capítulo III, temos muitas regras."
        result = section_boundary_precision(
            answer,
            "CAPÍTULO I",
            ["Capítulo II", "Capítulo III"],
        )
        assert result == 0.0

    def test_partial_contamination(self):
        answer = "Conforme Capítulo II, temos regras importantes."
        result = section_boundary_precision(
            answer,
            "CAPÍTULO I",
            ["Capítulo II", "Capítulo III"],
        )
        assert result == 0.5

    def test_no_other_labels(self):
        answer = "Qualquer texto aqui."
        assert section_boundary_precision(answer, "CAPÍTULO I", []) == 1.0

    def test_case_insensitive_contamination(self):
        answer = "Regras do capítulo iii."
        result = section_boundary_precision(
            answer,
            "CAPÍTULO I",
            ["CAPÍTULO III"],
        )
        assert result == 0.0


# ── evaluate_structural_summary ───────────────────────────────────────────

class TestEvaluateStructuralSummary:
    def test_with_matching_summaries(self):
        summaries = [
            {
                "node_type": "capitulo",
                "numeral": "II",
                "label": "CAPÍTULO II - DOS OBJETIVOS",
                "artigos_cobertos": ["Art. 3", "Art. 4"],
                "resumo_executivo": "Trata dos objetivos sociais.",
            },
        ]
        dataset = [
            {
                "question": "Resuma o Capítulo II",
                "target_type": "capitulo",
                "target_numeral": "II",
                "expected_articles": ["Art. 3", "Art. 4"],
                "query_intent": "summary_structural",
            },
        ]
        result = evaluate_structural_summary(summaries, dataset=dataset)
        assert result["queries_evaluated"] == 1
        assert result["structural_hit_at_1"] == 1.0
        assert result["article_coverage"] == 1.0
        assert result["summary_found_rate"] == 1.0
        assert result["summary_valid_rate"] == 1.0
        assert len(result["per_query"]) == 1

    def test_with_no_matching_summaries(self):
        summaries = [
            {
                "node_type": "capitulo",
                "numeral": "V",
                "label": "CAPÍTULO V - DAS SANÇÕES",
                "artigos_cobertos": [],
                "resumo_executivo": "Sanções.",
            },
        ]
        dataset = [
            {
                "question": "Resuma o Capítulo II",
                "target_type": "capitulo",
                "target_numeral": "II",
                "expected_articles": [],
                "query_intent": "summary_structural",
            },
        ]
        result = evaluate_structural_summary(summaries, dataset=dataset)
        assert result["structural_hit_at_1"] == 0.0
        pq = result["per_query"][0]
        assert pq["matched"] == ""

    def test_empty_summaries(self):
        dataset = [
            {
                "question": "Resuma o Capítulo II",
                "target_type": "capitulo",
                "target_numeral": "II",
                "expected_articles": [],
                "query_intent": "summary_structural",
            },
        ]
        result = evaluate_structural_summary([], dataset=dataset)
        assert result["structural_hit_at_1"] == 0.0
        assert result["article_coverage"] == 0.0

    def test_multiple_queries(self):
        summaries = [
            {
                "node_type": "capitulo",
                "numeral": "I",
                "label": "CAPÍTULO I - DAS OBRIGAÇÕES",
                "artigos_cobertos": ["Art. 1"],
                "resumo_executivo": "Obrigações.",
            },
            {
                "node_type": "secao",
                "numeral": "III",
                "label": "SEÇÃO III - DAS PENALIDADES",
                "artigos_cobertos": [],
                "resumo_executivo": "Penalidades.",
            },
        ]
        dataset = [
            {
                "question": "Resuma o Capítulo I",
                "target_type": "capitulo",
                "target_numeral": "I",
                "expected_articles": ["Art. 1"],
                "query_intent": "summary_structural",
            },
            {
                "question": "Explique a Seção III",
                "target_type": "secao",
                "target_numeral": "III",
                "expected_articles": [],
                "query_intent": "summary_structural",
            },
        ]
        result = evaluate_structural_summary(summaries, dataset=dataset)
        assert result["queries_evaluated"] == 2
        # Both should match
        assert result["structural_hit_at_1"] == 1.0

    def test_partial_coverage_in_evaluation(self):
        summaries = [
            {
                "node_type": "capitulo",
                "numeral": "I",
                "label": "CAPÍTULO I",
                "artigos_cobertos": ["Art. 1"],
                "resumo_executivo": "Teste.",
            },
        ]
        dataset = [
            {
                "question": "Resuma o Capítulo I",
                "target_type": "capitulo",
                "target_numeral": "I",
                "expected_articles": ["Art. 1", "Art. 2"],
                "query_intent": "summary_structural",
            },
        ]
        result = evaluate_structural_summary(summaries, dataset=dataset)
        assert result["article_coverage"] == 0.5

    def test_fallback_success_rate(self):
        summaries = [
            {
                "node_type": "capitulo",
                "numeral": "II",
                "label": "CAPITULO II",
                "artigos_cobertos": [],
                "resumo_executivo": "Resumo",
                "status": "fallback_only",
            },
        ]
        dataset = [
            {
                "question": "Resuma o Capitulo II",
                "target_type": "capitulo",
                "target_numeral": "2",
                "expected_articles": [],
                "query_intent": "summary_structural",
            },
        ]
        result = evaluate_structural_summary(summaries, dataset=dataset)
        assert result["fallback_success_rate"] == 1.0
