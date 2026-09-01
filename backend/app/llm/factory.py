import os

from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider


def get_llm_provider(model: str | None = None) -> LLMProvider:
    """model overrides the general RELEASEFORGE_LLM_MODEL default for this one call —
    used by review() to run a different, specifically-evaluated model for that role
    (see graph.py's DEFAULT_REVIEW_MODEL) without changing every other node's model."""
    provider = os.getenv("RELEASEFORGE_LLM_PROVIDER", "ollama")

    if provider == "ollama":
        resolved_model = model or os.getenv("RELEASEFORGE_LLM_MODEL", "qwen2.5:7b-instruct")
        embedding_model = os.getenv("RELEASEFORGE_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaProvider(model=resolved_model, embedding_model=embedding_model)

    if provider == "openai":
        resolved_model = model or os.getenv("RELEASEFORGE_LLM_MODEL", "gpt-4o-mini")
        embedding_model = os.getenv("RELEASEFORGE_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIProvider(model=resolved_model, embedding_model=embedding_model)

    # anthropic provider gets added here later, same LLMProvider shape.
    raise ValueError(f"Unknown LLM provider: {provider}")
