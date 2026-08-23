import os

from .base import LLMProvider
from .ollama_provider import OllamaProvider


def get_llm_provider() -> LLMProvider:
    provider = os.getenv("RELEASEFORGE_LLM_PROVIDER", "ollama")

    if provider == "ollama":
        model = os.getenv("RELEASEFORGE_LLM_MODEL", "qwen2.5:7b-instruct")
        embedding_model = os.getenv("RELEASEFORGE_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaProvider(model=model, embedding_model=embedding_model)

    # openai / anthropic providers get added here later, same LLMProvider shape.
    raise ValueError(f"Unknown LLM provider: {provider}")
