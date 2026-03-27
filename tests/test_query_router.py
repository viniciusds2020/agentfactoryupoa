from src.query_router import detect_query_intent


def test_router_returns_vector_when_no_structured_data():
    intent = detect_query_intent("codigo 10101039 precisa autorizacao?", collection_has_structured=False)
    assert intent.route == "vector"


def test_router_detects_structured_by_codigo():
    intent = detect_query_intent("codigo 10101039 precisa autorizacao?", collection_has_structured=True)
    assert intent.route == "structured"
    assert intent.structured_filters["codigo"] == "10101039"


def test_router_detects_structured_by_numeric_id():
    intent = detect_query_intent("procedimento 10101039", collection_has_structured=True)
    assert intent.route == "structured"
    assert intent.structured_filters["codigo"] == "10101039"


def test_router_detects_hybrid_by_field_value():
    intent = detect_query_intent("quais procedimentos sao de emergencia sim?", collection_has_structured=True)
    assert intent.route == "hybrid"
    assert intent.structured_filters["emergencia"] == "sim"


def test_router_falls_back_to_vector_for_natural_language():
    intent = detect_query_intent("qual o prazo de carencia para exames?", collection_has_structured=True)
    assert intent.route == "vector"
