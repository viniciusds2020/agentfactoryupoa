"""Text normalization and tokenization utilities for PT-BR."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from src.utils import get_logger

logger = get_logger(__name__)

# Stopwords PT-BR (já normalizadas: sem acento, lowercase)
_PT_STOPWORDS: set[str] = {
    "a", "ao", "aos", "as", "ate", "com", "como", "da", "das", "de", "del",
    "dem", "des", "do", "dos", "e", "eh", "ela", "elas", "ele", "eles", "em",
    "entre", "era", "essa", "essas", "esse", "esses", "esta", "estas", "este",
    "estes", "eu", "foi", "for", "foram", "ha", "isso", "isto", "ja", "la",
    "lhe", "lhes", "lo", "los", "mais", "mas", "me", "meu", "meus", "minha",
    "minhas", "muito", "na", "nao", "nas", "nem", "no", "nos", "nossa",
    "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quais", "quando",
    "que", "quem", "sao", "se", "sem", "ser", "sera", "seu", "seus", "sim",
    "so", "sob", "sobre", "sua", "suas", "tal", "tambem", "te", "tem", "ter",
    "toda", "todas", "todo", "todos", "tu", "tua", "tuas", "tudo", "um",
    "uma", "umas", "uns", "vai", "vao", "voce", "voces",
}


@lru_cache(maxsize=1)
def _get_stemmer():
    """Lazy-load RSLP stemmer (downloads ~10KB data on first use)."""
    import nltk
    try:
        nltk.data.find("stemmers/rslp")
    except LookupError:
        nltk.download("rslp", quiet=True)
    from nltk.stem import RSLPStemmer
    return RSLPStemmer()


_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_RE = re.compile(
    r"\b(m{0,3})(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})\b",
    re.IGNORECASE,
)


def _roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to integer. Returns 0 if invalid."""
    s = s.lower()
    if not s or not all(c in _ROMAN_VALUES for c in s):
        return 0
    total = 0
    prev = 0
    for c in reversed(s):
        val = _ROMAN_VALUES[c]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def _replace_roman_numerals(text: str) -> str:
    """Replace Roman numerals with Arabic equivalents in text.

    Only replaces standalone words that are valid Roman numerals (I-MMMCMXCIX).
    """
    def _replace(match: re.Match) -> str:
        token = match.group(0)
        value = _roman_to_int(token)
        return str(value) if value > 0 else token

    return _ROMAN_RE.sub(_replace, text)


_INT_TO_ROMAN = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]

_ARABIC_IN_QUERY_RE = re.compile(r"\b(\d{1,4})\b")


def _int_to_roman(n: int) -> str:
    """Convert integer (1-3999) to Roman numeral string."""
    if n < 1 or n > 3999:
        return str(n)
    result = []
    for value, numeral in _INT_TO_ROMAN:
        while n >= value:
            result.append(numeral)
            n -= value
    return "".join(result)


def normalize_query_numerals(query: str) -> str:
    """Expand a query so it contains both Arabic and Roman forms of numbers.

    Examples:
        "capitulo 1"  -> "capitulo 1 I"
        "artigo IV"   -> "artigo IV 4"
        "seção 10"    -> "seção 10 X"
    """
    additions: list[str] = []

    # Arabic -> add Roman form
    for m in _ARABIC_IN_QUERY_RE.finditer(query):
        n = int(m.group(1))
        if 1 <= n <= 3999:
            additions.append(_int_to_roman(n))

    # Roman -> add Arabic form
    for m in _ROMAN_RE.finditer(query):
        val = _roman_to_int(m.group(0))
        if val > 0:
            additions.append(str(val))

    if additions:
        return query + " " + " ".join(additions)
    return query


def _normalize(text: str) -> str:
    """Lowercase, remove accents, replace Roman numerals, keep alphanumeric + spaces."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _replace_roman_numerals(text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text


def tokenize(text: str, use_stemming: bool = True) -> list[str]:
    """Tokenize with stopword removal and optional RSLP stemming for PT-BR."""
    tokens = _normalize(text).split()
    tokens = [t for t in tokens if t not in _PT_STOPWORDS and len(t) > 1]
    if use_stemming:
        stemmer = _get_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


