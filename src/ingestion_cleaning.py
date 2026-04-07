"""Text normalization helpers for ingestion."""
from __future__ import annotations

import re
import unicodedata

# Abreviacoes PT-BR comuns que nao devem ser tratadas como fim de frase
_ABBREVIATIONS = re.compile(
    r"\b(Art|Inc|Sr|Sra|Dr|Dra|Prof|Ltda|S\.A|Jr|n|p|fl|fls|vol|ed|etc|ex)\.",
    re.IGNORECASE,
)
_ABBREV_PLACEHOLDER = "\x00"

_PAGE_BREAK_MARKER = "\n<!-- PAGE_BREAK -->\n"
_PAGE_BREAK_RE = re.compile(r"<!-- PAGE_BREAK -->")

_PAGE_HEADER_PATTERNS = [
    re.compile(
        r"^[^\n]{0,200}(?:Nº/Rev\.:|N°/Rev\.:)[^\n]*Este documento faz parte[^\n]*\n?",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(r"\(Baixado por [^\n)]+\)\s*", re.IGNORECASE),
    re.compile(r"Página\s+\d+\s+de\s+\d+\s*", re.IGNORECASE),
    re.compile(r"Classificação da Informação:\s*\w+\s*", re.IGNORECASE),
    re.compile(r"<!--\s*image\s*-->\s*", re.IGNORECASE),
]

_HYPHENATED_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_BROKEN_LINE_RE = re.compile(r"(?<=[a-záàâãéèêíóôõúüç,;])\s*\n\s*(?=[a-záàâãéèêíóôõúüç])", re.IGNORECASE)
_REPEATED_BLOCK_MIN_LEN = 40
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _strip_page_headers(text: str) -> str:
    for pattern in _PAGE_HEADER_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _merge_broken_lines(text: str) -> str:
    text = _HYPHENATED_BREAK_RE.sub(r"\1\2", text)
    text = _BROKEN_LINE_RE.sub(" ", text)
    return text


def _deduplicate_paragraphs(text: str) -> str:
    lines = text.split("\n\n")
    seen: set[str] = set()
    result: list[str] = []
    for block in lines:
        stripped = block.strip()
        if len(stripped) < _REPEATED_BLOCK_MIN_LEN:
            result.append(block)
            continue
        key = re.sub(r"\s+", " ", stripped.lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(block)
    return "\n\n".join(result)


def _remove_orphan_page_numbers(text: str) -> str:
    lines = text.splitlines()
    result = [line for line in lines if not re.fullmatch(r"\s*\d{1,4}\s*", line)]
    return "\n".join(result)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


def deep_clean_text(text: str) -> str:
    text = _merge_broken_lines(text)
    text = _remove_orphan_page_numbers(text)
    text = _deduplicate_paragraphs(text)
    text = _normalize_whitespace(text)
    return text


def _normalize_markdown(text: str) -> str:
    return _MARKDOWN_HEADING_RE.sub("", text)


def _normalize_structural_headers(text: str) -> str:
    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith(("[", "("))
            and stripped.endswith(("]", ")"))
            and any(token in stripped.upper() for token in ("CAP", "TIT", "SE"))
        ):
            normalized_lines.append(stripped[1:-1].strip())
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _normalize_heading_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized.upper().strip()


def _split_sentences(text: str) -> list[str]:
    protected = _ABBREVIATIONS.sub(lambda m: m.group(0).replace(".", _ABBREV_PLACEHOLDER), text)
    raw_sentences = re.split(r"(?<=[.!?;])\s+", protected)
    return [s.replace(_ABBREV_PLACEHOLDER, ".").strip() for s in raw_sentences if s.strip()]


def _split(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if sent_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            start = 0
            while start < sent_len:
                chunks.append(sent[start : start + chunk_size].strip())
                start += chunk_size - chunk_overlap
            continue

        if current_len + sent_len + (1 if current else 0) > chunk_size:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + (1 if overlap else 0) > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + (1 if len(overlap) > 1 else 0)
            current = overlap
            current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)

        current.append(sent)
        current_len += sent_len + (1 if len(current) > 1 else 0)

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c]
