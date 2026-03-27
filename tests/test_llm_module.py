import sys
import types

from src import llm


class _FakeRateLimitError(Exception):
    """Fake RateLimitError for groq mock modules."""
    pass


class _FakeAPIStatusError(Exception):
    """Fake APIStatusError for groq mock modules."""
    def __init__(self, message="", status_code=500):
        super().__init__(message)
        self.status_code = status_code


def _groq_module(**extras):
    """Build a fake groq module namespace with RateLimitError and APIStatusError."""
    ns = {"RateLimitError": _FakeRateLimitError, "APIStatusError": _FakeAPIStatusError}
    ns.update(extras)
    return types.SimpleNamespace(**ns)


def test_embed_uses_fastembed_backend(monkeypatch):
    monkeypatch.setattr("src.llm.get_settings", lambda: type("S", (), {"embedding_batch_size": 8, "embedding_model": "unused"})())

    class FakeVector:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class FakeFastEmbedModel:
        def embed(self, texts):
            return [FakeVector([1.0, 2.0]) for _ in texts]

    monkeypatch.setattr("src.llm._load_fastembed", lambda model_name: FakeFastEmbedModel())

    result = llm.embed(["abc"], model_name="BAAI/bge-m3")
    assert result == [[1.0, 2.0]]


def test_embed_uses_sentence_transformers_backend(monkeypatch):
    monkeypatch.setattr("src.llm.get_settings", lambda: type("S", (), {"embedding_batch_size": 4, "embedding_model": "unused"})())

    class FakeSentenceModel:
        def encode(self, texts, normalize_embeddings, show_progress_bar, batch_size):
            assert normalize_embeddings is True
            assert show_progress_bar is False
            assert batch_size == 4

            class FakeVector:
                def __init__(self, values):
                    self._values = values

                def tolist(self):
                    return self._values

            return [FakeVector([0.3, 0.4]) for _ in texts]

    monkeypatch.setattr("src.llm._load_st_model", lambda model_name: FakeSentenceModel())

    result = llm.embed(["abc"], model_name="rufimelo/Legal-BERTimbau-sts-large")
    assert result == [[0.3, 0.4]]


def test_embedding_dimension_uses_override(monkeypatch):
    monkeypatch.setattr("src.llm.embed", lambda texts, model_name=None: [[0.1, 0.2, 0.3]])
    assert llm.embedding_dimension("model-x") == 3


def test_chat_routes_to_groq(monkeypatch):
    monkeypatch.setattr("src.llm.get_settings", lambda: type("S", (), {"llm_provider": "groq"})())
    monkeypatch.setattr("src.llm._chat_groq", lambda messages, system, settings: "groq-ok")
    assert llm.chat([{"role": "user", "content": "oi"}]) == "groq-ok"


def test_chat_routes_to_anthropic(monkeypatch):
    monkeypatch.setattr("src.llm.get_settings", lambda: type("S", (), {"llm_provider": "anthropic"})())
    monkeypatch.setattr("src.llm._chat_anthropic", lambda messages, system, settings: "anthropic-ok")
    assert llm.chat([{"role": "user", "content": "oi"}]) == "anthropic-ok"


def test_clear_embed_cache_clears_both_caches(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(llm._load_fastembed, "cache_clear", lambda: calls.append("fastembed"))
    monkeypatch.setattr(llm._load_st_model, "cache_clear", lambda: calls.append("st"))

    llm.clear_embed_cache()

    assert calls == ["fastembed", "st"]


def test_chat_groq_builds_full_message_list(monkeypatch):
    captured: dict = {}

    class FakeGroq:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="resposta-groq"))]
            )

    monkeypatch.setitem(sys.modules, "groq", _groq_module(Groq=FakeGroq))
    settings = type("S", (), {"groq_api_key": "groq-key", "llm_model": "llama-x"})()

    result = llm._chat_groq([{"role": "user", "content": "oi"}], "sistema", settings)

    assert result == "resposta-groq"
    assert captured["api_key"] == "groq-key"
    assert captured["model"] == "llama-x"
    assert captured["messages"][0] == {"role": "system", "content": "sistema"}
    assert captured["messages"][1] == {"role": "user", "content": "oi"}


def test_chat_groq_returns_empty_string_when_provider_content_is_none(monkeypatch):
    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kwargs: types.SimpleNamespace(
                        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None))]
                    )
                )
            )

    monkeypatch.setitem(sys.modules, "groq", _groq_module(Groq=FakeGroq))
    settings = type("S", (), {"groq_api_key": "groq-key", "llm_model": "llama-x"})()

    assert llm._chat_groq([], "", settings) == ""


def test_chat_anthropic_passes_system_and_messages(monkeypatch):
    captured: dict = {}

    class FakeAnthropicClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = types.SimpleNamespace(create=self._create)

        def _create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(text="resposta-anthropic")])

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=FakeAnthropicClient),
    )
    settings = type("S", (), {"anthropic_api_key": "anth-key", "llm_model": "claude-x", "llm_provider": "anthropic"})()

    result = llm._chat_anthropic([{"role": "user", "content": "oi"}], "contexto", settings)

    assert result == "resposta-anthropic"
    assert captured["api_key"] == "anth-key"
    assert captured["model"] == "claude-x"
    assert captured["system"] == "contexto"
    assert captured["messages"] == [{"role": "user", "content": "oi"}]


def test_groq_fallback_returns_alternative_model():
    settings = type("S", (), {"llm_model": "llama-3.3-70b-versatile"})()
    result = llm._groq_fallback(settings, "rate limit")
    assert result == ("groq", "meta-llama/llama-4-scout-17b-16e-instruct")


def test_groq_fallback_skips_current_model():
    settings = type("S", (), {"llm_model": "meta-llama/llama-4-scout-17b-16e-instruct"})()
    result = llm._groq_fallback(settings, "rate limit")
    assert result == ("groq", "llama-3.1-8b-instant")


def test_groq_fallback_returns_none_when_all_exhausted():
    settings = type("S", (), {"llm_model": "llama-3.3-70b-versatile"})()
    all_fallbacks = {"meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"}
    result = llm._groq_fallback(settings, "rate limit", exclude=all_fallbacks)
    assert result is None


def test_chat_groq_retries_with_fallback_model_on_rate_limit(monkeypatch):
    call_count = {"n": 0}

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _FakeRateLimitError("rate limit")
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="fallback-ok"))]
            )

    monkeypatch.setitem(sys.modules, "groq", _groq_module(Groq=FakeGroq))
    settings = type("S", (), {
        "groq_api_key": "key", "llm_model": "llama-3.3-70b-versatile", "anthropic_api_key": ""
    })()

    result = llm._chat_groq([{"role": "user", "content": "oi"}], "", settings)
    assert result == "fallback-ok"
    assert call_count["n"] == 2
