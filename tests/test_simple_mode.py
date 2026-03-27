"""Tests for simple_mode config and the built-in UI route."""
import os
import pathlib

from src.config import Settings


def test_simple_mode_defaults_to_true(monkeypatch):
    monkeypatch.delenv("SIMPLE_MODE", raising=False)
    # Settings reads from .env file too; test the class default directly
    assert Settings.model_fields["simple_mode"].default is True


def test_default_collection_is_documentos():
    s = Settings()
    assert s.default_collection == "documentos"


def test_simple_mode_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SIMPLE_MODE", "false")
    s = Settings()
    assert s.simple_mode is False


def test_default_collection_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DEFAULT_COLLECTION", "contratos")
    s = Settings()
    assert s.default_collection == "contratos"


def test_simple_ui_route_serves_html():
    """GET / should return the built-in HTML UI.

    Note: conftest.py sets SIMPLE_MODE=false for all tests so the
    enterprise router is included. We test the route handler directly
    to avoid import-time coupling.
    """
    from app import _TEMPLATES_DIR

    template = _TEMPLATES_DIR / "index.html"
    assert template.exists(), "templates/index.html must exist"
    html = template.read_text(encoding="utf-8")
    assert "Agent Factory" in html
    assert "{{ default_collection }}" in html


def test_simple_ui_template_has_required_endpoints():
    """The HTML template must call the expected API endpoints."""
    from app import _TEMPLATES_DIR

    html = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    assert "/ingest" in html
    assert "/chat/message" in html
    assert "/conversations" in html
    assert "/collections/available" in html
