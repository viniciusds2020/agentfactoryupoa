from src.evaluation import evaluate_tabular_plans


def test_evaluate_tabular_plans_reports_summary():
    def planner(question: str) -> dict | None:
        question = question.lower()
        if "media de idade" in question:
            return {
                "intent": "tabular_aggregate",
                "metric_column": "idade",
                "filters": [{"column": "estado", "value": "CE"}],
                "expected_unit": "anos",
            }
        if "renda total" in question:
            return {
                "intent": "tabular_aggregate",
                "metric_column": "renda_mensal",
                "filters": [{"column": "estado", "value": "SP"}],
                "expected_unit": "brl",
            }
        if "estados uf" in question:
            return {"intent": "tabular_distinct", "filters": []}
        if "colunas da tabela" in question:
            return {"intent": "tabular_schema", "filters": []}
        return None

    report = evaluate_tabular_plans(planner)

    assert report["cases"] >= 4
    assert report["summary"]["tabular_plan_success_rate"] == 1.0
    assert report["summary"]["unit_render_accuracy"] == 1.0
