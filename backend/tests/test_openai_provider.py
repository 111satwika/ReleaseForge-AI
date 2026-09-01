"""Fast, deterministic tests for OpenAIProvider's construction and the LLM factory's
provider selection — no real API calls (that's verified live, not here; a live call
against a real key was confirmed manually before this was wired in). Also covers
_resolve_review_model()'s three branches, since that logic exists specifically to stop
the Review Agent's Ollama-only DEFAULT_REVIEW_MODEL from leaking into a non-Ollama
provider and breaking every review() call with a nonexistent model name."""

import pytest

from app.agents.graph import DEFAULT_REVIEW_MODEL, _resolve_review_model
from app.llm.factory import get_llm_provider
from app.llm.openai_provider import OpenAIProvider


def test_openai_provider_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIProvider()


def test_openai_provider_accepts_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    provider = OpenAIProvider()
    assert provider.model == "gpt-4o-mini"
    assert provider.embedding_model == "text-embedding-3-small"


def test_openai_provider_accepts_key_passed_directly():
    provider = OpenAIProvider(api_key="sk-test-not-real", model="gpt-4o")
    assert provider.model == "gpt-4o"


def test_factory_selects_openai_provider(monkeypatch):
    monkeypatch.setenv("RELEASEFORGE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    provider = get_llm_provider()
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o-mini"


def test_factory_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("RELEASEFORGE_LLM_PROVIDER", "not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider()


def test_resolve_review_model_explicit_override_always_wins(monkeypatch):
    monkeypatch.setenv("RELEASEFORGE_REVIEW_MODEL", "gpt-4o")
    monkeypatch.setenv("RELEASEFORGE_LLM_PROVIDER", "ollama")
    assert _resolve_review_model() == "gpt-4o"


def test_resolve_review_model_defaults_to_golden_set_model_under_ollama(monkeypatch):
    monkeypatch.delenv("RELEASEFORGE_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("RELEASEFORGE_LLM_PROVIDER", raising=False)
    assert _resolve_review_model() == DEFAULT_REVIEW_MODEL


def test_resolve_review_model_defers_to_provider_default_under_openai(monkeypatch):
    """The real bug this exists to prevent: without this, review() would pass
    'qwen2.5-coder:7b' straight to the OpenAI API and every review call would fail."""
    monkeypatch.delenv("RELEASEFORGE_REVIEW_MODEL", raising=False)
    monkeypatch.setenv("RELEASEFORGE_LLM_PROVIDER", "openai")
    assert _resolve_review_model() is None
