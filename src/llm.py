"""LLM and embedding calls.

Chat providers (LLM_PROVIDER no .env):
  groq      -> llama-3.3-70b-versatile, llama-4-scout-17b-16e-instruct, llama-3.1-8b-instant, mixtral-8x7b-32768
  anthropic -> claude-sonnet-4-6, claude-haiku-4-5-20251001

Embeddings use two backends with automatic selection by model name:
  fastembed             -> lightweight multilingual ONNX models
  sentence-transformers -> specialized PT-BR models (for example Legal-BERTimbau)
"""
from __future__ import annotations

from functools import lru_cache

from src.config import get_settings
from src.utils import get_logger

logger = get_logger(__name__)

_FASTEMBED_MODELS: set[str] = {
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
}


@lru_cache(maxsize=4)
def _load_fastembed(model_name: str):  # type: ignore[return]
    from fastembed import TextEmbedding

    logger.info(f"[fastembed] Loading '{model_name}'")
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=4)
def _load_st_model(model_name: str):  # type: ignore[return]
    from sentence_transformers import SentenceTransformer

    logger.info(f"[sentence-transformers] Loading '{model_name}'")
    return SentenceTransformer(model_name, device="cpu")


def embed(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed texts using the configured model or an explicit override."""
    from src.observability import EMBEDDING_DURATION
    from time import perf_counter

    settings = get_settings()
    model_name = model_name or settings.embedding_model

    start = perf_counter()
    if model_name in _FASTEMBED_MODELS:
        model = _load_fastembed(model_name)
        result = [v.tolist() for v in model.embed(texts)]
    else:
        model = _load_st_model(model_name)
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=settings.embedding_batch_size,
        )
        result = [v.tolist() for v in vectors]

    EMBEDDING_DURATION.labels(model=model_name).observe(perf_counter() - start)
    return result


def embedding_dimension(model_name: str | None = None) -> int:
    """Resolve embedding dimension at runtime."""
    model_name = model_name or get_settings().embedding_model
    dim = len(embed(["dimensão"], model_name=model_name)[0])
    logger.info(f"Embedding dimension for '{model_name}': {dim}")
    return dim


def clear_embed_cache() -> None:
    """Clear cached embedding models."""
    _load_fastembed.cache_clear()
    _load_st_model.cache_clear()
    logger.info("Embedding model cache cleared")


_GROQ_FALLBACK_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


def _groq_fallback(settings, error_msg: str, *, exclude: set[str] | None = None):
    """Determine fallback strategy when Groq returns 429/413.

    Returns (provider, model) tuple or None.
    """
    exclude = exclude or set()
    for model in _GROQ_FALLBACK_MODELS:
        if model != settings.llm_model and model not in exclude:
            logger.warning(f"Groq error on '{settings.llm_model}', falling back to '{model}'")
            return ("groq", model)

    return None


def _is_groq_retryable(exc) -> bool:
    """Check if a Groq exception should trigger fallback (rate limit or payload too large)."""
    from groq import RateLimitError, APIStatusError
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (413, 429):
        return True
    return False


def _truncate_messages(messages: list[dict], max_chars: int = 18000) -> list[dict]:
    """Truncate message content to fit within a character budget.

    Preserves the system message and the last user message. Trims context
    from the middle (older history and long assistant messages) first.
    """
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= max_chars:
        return messages

    # Strategy: keep system + last message intact, trim everything in between
    result = list(messages)
    # Find system and last user message indices
    system_idx = 0 if result and result[0]["role"] == "system" else -1
    last_user_idx = -1
    for i in range(len(result) - 1, -1, -1):
        if result[i]["role"] == "user":
            last_user_idx = i
            break

    # Trim middle messages (history) by halving their content
    protected = {system_idx, last_user_idx}
    budget_remaining = max_chars
    for idx in protected:
        if idx >= 0:
            budget_remaining -= len(result[idx].get("content", ""))

    trimmed = []
    for i, msg in enumerate(result):
        if i in protected:
            trimmed.append(msg)
            continue
        content = msg.get("content", "")
        # Allocate proportional budget to each middle message
        middle_count = len(result) - len(protected)
        per_msg_budget = max(200, budget_remaining // max(middle_count, 1))
        if len(content) > per_msg_budget:
            trimmed.append({**msg, "content": content[:per_msg_budget] + "...[truncado]"})
        else:
            trimmed.append(msg)
            budget_remaining -= len(content)

    return trimmed


def chat(messages: list[dict], system: str = "") -> str:
    from src.observability import LLM_CALLS_TOTAL, LLM_DURATION
    from time import perf_counter

    settings = get_settings()
    start = perf_counter()
    if settings.llm_provider == "groq":
        result = _chat_groq(messages, system, settings)
    else:
        result = _chat_anthropic(messages, system, settings)

    LLM_CALLS_TOTAL.labels(provider=settings.llm_provider, model=settings.llm_model).inc()
    LLM_DURATION.labels(provider=settings.llm_provider).observe(perf_counter() - start)
    return result


def _chat_groq(messages: list[dict], system: str, settings) -> str:
    from groq import Groq, RateLimitError, APIStatusError

    client = Groq(api_key=settings.groq_api_key)
    full: list[dict] = []
    if system:
        full.append({"role": "system", "content": system})
    full.extend(messages)

    tried: set[str] = {settings.llm_model}

    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=full,  # type: ignore[arg-type]
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except (RateLimitError, APIStatusError) as exc:
        if not _is_groq_retryable(exc):
            raise
        fallback = _groq_fallback(settings, str(exc), exclude=tried)
        if fallback is None:
            raise
        _, model = fallback
        tried.add(model)
        truncated = _truncate_messages(full)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=truncated,  # type: ignore[arg-type]
                max_tokens=2048,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except (RateLimitError, APIStatusError) as exc2:
            if not _is_groq_retryable(exc2):
                raise
            fallback2 = _groq_fallback(settings, str(exc2), exclude=tried)
            if fallback2 is None:
                raise
            _, model2 = fallback2
            tried.add(model2)
            response = client.chat.completions.create(
                model=model2,
                messages=_truncate_messages(full, max_chars=12000),  # type: ignore[arg-type]
                max_tokens=2048,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""


def _chat_anthropic(messages: list[dict], system: str, settings) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model if settings.llm_provider == "anthropic" else "claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    return response.content[0].text  # type: ignore[index]


# ── Streaming ──────────────────────────────────────────────────────────────


def chat_stream(messages: list[dict], system: str = ""):
    """Yield text chunks from the LLM as a generator (for SSE)."""
    settings = get_settings()
    if settings.llm_provider == "groq":
        yield from _chat_stream_groq(messages, system, settings)
    else:
        yield from _chat_stream_anthropic(messages, system, settings)


def _chat_stream_groq(messages: list[dict], system: str, settings):
    from groq import Groq, RateLimitError, APIStatusError

    client = Groq(api_key=settings.groq_api_key)
    full: list[dict] = []
    if system:
        full.append({"role": "system", "content": system})
    full.extend(messages)

    tried: set[str] = {settings.llm_model}

    try:
        stream = client.chat.completions.create(
            model=settings.llm_model,
            messages=full,  # type: ignore[arg-type]
            max_tokens=2048,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except (RateLimitError, APIStatusError) as exc:
        if not _is_groq_retryable(exc):
            raise
        fallback = _groq_fallback(settings, str(exc), exclude=tried)
        if fallback is None:
            raise
        _, model = fallback
        tried.add(model)
        truncated = _truncate_messages(full)
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=truncated,  # type: ignore[arg-type]
                max_tokens=2048,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except (RateLimitError, APIStatusError) as exc2:
            if not _is_groq_retryable(exc2):
                raise
            fallback2 = _groq_fallback(settings, str(exc2), exclude=tried)
            if fallback2 is None:
                raise
            _, model2 = fallback2
            tried.add(model2)
            stream = client.chat.completions.create(
                model=model2,
                messages=_truncate_messages(full, max_chars=12000),  # type: ignore[arg-type]
                max_tokens=2048,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content


def _chat_stream_anthropic(messages: list[dict], system: str, settings):
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    with client.messages.stream(
        model=settings.llm_model if settings.llm_provider == "anthropic" else "claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text
