"""Contextual compression — extract only relevant sentences from retrieved chunks."""
from __future__ import annotations

import re

from src.lexical import tokenize
from src.utils import get_logger, log_event

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")


def compress_extractive(
    query: str,
    text: str,
    max_sentences: int = 3,
) -> str:
    """Extract the most relevant sentences from a chunk using token overlap scoring.

    Splits text into sentences, scores each by query token overlap, and returns
    the top-N sentences in their original order.
    """
    sentences = _SENTENCE_SPLIT.split(text.strip())
    if len(sentences) <= max_sentences:
        return text

    query_tokens = set(tokenize(query))
    scored = []
    for i, sent in enumerate(sentences):
        sent_tokens = set(tokenize(sent))
        overlap = len(query_tokens & sent_tokens)
        scored.append((overlap, i, sent))

    # Sort by score descending, take top-N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_sentences]

    # Return in original order
    top.sort(key=lambda x: x[1])
    return " ".join(s for _, _, s in top)


def compress_chunks(
    query: str,
    chunks: list[dict],
    method: str = "extractive",
    max_sentences: int = 3,
) -> list[dict]:
    """Compress a list of retrieved chunks, extracting only relevant portions.

    Chunks with ``chunk_type == "parent"`` are NOT compressed (legal articles
    need full integrity).

    Args:
        query: The user's question.
        chunks: Retrieved chunks (each must have "text" and optionally "metadata").
        method: "extractive" (default) for token-overlap based extraction.
        max_sentences: Max sentences to keep per chunk (extractive only).

    Returns:
        Chunks with compressed text.
    """
    if method != "extractive":
        log_event(logger, 30, "Unknown compression method, skipping", method=method)
        return chunks

    compressed = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        # Don't compress parent chunks (legal articles need full context)
        if meta.get("chunk_type") == "parent":
            compressed.append(chunk)
            continue

        original_text = chunk["text"]
        new_text = compress_extractive(query, original_text, max_sentences)
        new_chunk = {**chunk, "text": new_text}
        compressed.append(new_chunk)

    log_event(
        logger, 20, "Contextual compression done",
        method=method,
        input_chunks=len(chunks),
        max_sentences=max_sentences,
    )
    return compressed
