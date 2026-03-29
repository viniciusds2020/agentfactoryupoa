import pytest
from src.lexical import tokenize, _PT_STOPWORDS, _normalize, roman_to_int, _replace_roman_numerals, normalize_query_numerals, int_to_roman


def test_tokenize_removes_accents():
    tokens = tokenize("rescisao contratual", use_stemming=False)
    assert "rescisao" in tokens
    assert "contratual" in tokens


def test_tokenize_lowercases():
    tokens = tokenize("CLT FGTS", use_stemming=False)
    assert "clt" in tokens
    assert "fgts" in tokens


def test_tokenize_removes_stopwords():
    tokens = tokenize("rescisao de contrato para o funcionario", use_stemming=False)
    assert "de" not in tokens
    assert "para" not in tokens
    assert "rescisao" in tokens
    assert "contrato" in tokens
    assert "funcionario" in tokens


def test_tokenize_applies_stemming():
    stem_contratos = tokenize("contratos", use_stemming=True)
    stem_contratual = tokenize("contratual", use_stemming=True)
    assert stem_contratos[0] == stem_contratual[0]


def test_tokenize_without_stemming():
    tokens = tokenize("contratos contratual", use_stemming=False)
    assert "contratos" in tokens
    assert "contratual" in tokens


def test_stopwords_are_normalized():
    for sw in _PT_STOPWORDS:
        assert sw == _normalize(sw).strip(), f"Stopword '{sw}' is not normalized"


# ── Roman numeral tests ─────────────────────────────────────────────────────


def testroman_to_int_basic():
    assert roman_to_int("I") == 1
    assert roman_to_int("V") == 5
    assert roman_to_int("X") == 10
    assert roman_to_int("L") == 50
    assert roman_to_int("C") == 100
    assert roman_to_int("D") == 500
    assert roman_to_int("M") == 1000


def testroman_to_int_compound():
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("XIV") == 14
    assert roman_to_int("XL") == 40
    assert roman_to_int("XLII") == 42
    assert roman_to_int("XC") == 90
    assert roman_to_int("CXXIII") == 123
    assert roman_to_int("CD") == 400
    assert roman_to_int("CM") == 900
    assert roman_to_int("MMXXVI") == 2026


def testroman_to_int_invalid():
    assert roman_to_int("") == 0
    assert roman_to_int("ABC") == 0


def test_replace_roman_numerals_in_text():
    assert _replace_roman_numerals("Art. V da CLT") == "Art. 5 da CLT"
    assert _replace_roman_numerals("capitulo III secao IV") == "capitulo 3 secao 4"
    assert _replace_roman_numerals("inciso XIV") == "inciso 14"


def test_replace_roman_numerals_preserves_normal_words():
    text = "direito civil e criminal"
    assert _replace_roman_numerals(text) == text


def test_normalize_converts_roman_to_arabic():
    assert "5" in _normalize("Art. V")
    assert "14" in _normalize("Inciso XIV")


# ── Integer to Roman conversion ───────────────────────────────────────────


def testint_to_roman_basic():
    assert int_to_roman(1) == "I"
    assert int_to_roman(4) == "IV"
    assert int_to_roman(5) == "V"
    assert int_to_roman(9) == "IX"
    assert int_to_roman(10) == "X"
    assert int_to_roman(14) == "XIV"
    assert int_to_roman(42) == "XLII"
    assert int_to_roman(100) == "C"
    assert int_to_roman(2026) == "MMXXVI"


def testint_to_roman_out_of_range():
    assert int_to_roman(0) == "0"
    assert int_to_roman(4000) == "4000"


# ── Query numeral normalization ───────────────────────────────────────────


def test_normalize_query_arabic_adds_roman():
    result = normalize_query_numerals("capitulo 1")
    assert "I" in result
    assert "1" in result


def test_normalize_query_roman_adds_arabic():
    result = normalize_query_numerals("capitulo IV")
    assert "4" in result
    assert "IV" in result


def test_normalize_query_larger_numbers():
    result = normalize_query_numerals("artigo 14")
    assert "XIV" in result


def test_normalize_query_no_numbers_unchanged():
    query = "rescisao contratual"
    assert normalize_query_numerals(query) == query


def test_normalize_query_mixed():
    result = normalize_query_numerals("secao 3 artigo VII")
    assert "III" in result  # 3 -> III
    assert "7" in result    # VII -> 7
