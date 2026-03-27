from src.config import Settings


def test_defaults_are_sane():
    s = Settings()
    assert s.environment == "development"
    assert s.retrieval_top_k > 0
    assert s.chroma_path != ""


def test_env_override(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "20")
    s = Settings()
    assert s.environment == "production"
    assert s.retrieval_top_k == 20


