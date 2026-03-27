from src.utils import chunk_id, mask_pii, new_request_id, request_id_var
import pytest


def test_mask_pii_replaces_cpf():
    assert mask_pii("CPF: 123.456.789-00") == "CPF: [CPF]"


def test_mask_pii_replaces_cpf_without_punctuation():
    assert mask_pii("CPF: 12345678900") == "CPF: [CPF]"


def test_mask_pii_replaces_cnpj():
    assert mask_pii("CNPJ: 12.345.678/0001-90") == "CNPJ: [CNPJ]"


def test_mask_pii_replaces_email():
    assert mask_pii("Email: joao@empresa.com.br") == "Email: [EMAIL]"


def test_mask_pii_replaces_phone():
    assert mask_pii("Tel: (11) 98765-4321") == "Tel: [TELEFONE]"


def test_mask_pii_leaves_normal_text_unchanged():
    text = "Qual o prazo do contrato numero 42?"
    assert mask_pii(text) == text


def test_mask_pii_handles_multiple_patterns():
    text = "CPF 123.456.789-00 email joao@teste.com"
    masked = mask_pii(text)
    assert "[CPF]" in masked
    assert "[EMAIL]" in masked
    assert "123.456.789-00" not in masked


def test_chunk_id_is_deterministic():
    a = chunk_id("col", "doc1", 0)
    b = chunk_id("col", "doc1", 0)
    assert a == b


def test_chunk_id_differs_by_index():
    assert chunk_id("col", "doc1", 0) != chunk_id("col", "doc1", 1)


def test_chunk_id_differs_by_collection():
    assert chunk_id("col_a", "doc1", 0) != chunk_id("col_b", "doc1", 0)


def test_new_request_id_is_unique():
    assert new_request_id() != new_request_id()


def test_request_id_contextvar_propagation():
    token = request_id_var.set("req-test-123")
    try:
        assert request_id_var.get() == "req-test-123"
    finally:
        request_id_var.reset(token)
