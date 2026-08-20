import os

from .base import LLMProvider
from .ollama_provider import OllamaProvider


def get_llm_provider() -> LLMProvider:
    provider = os.getenv("DEPLOYGENIE_LLM_PROVIDER", "ollama")

    if provider == "ollama":
        model = os.getenv("DEPLOYGENIE_LLM_MODEL", "qwen2.5:7b-instruct")
        return OllamaProvider(model=model)

    # openai / anthropic providers get added here later, same LLMProvider shape.
    raise ValueError(f"Unknown LLM provider: {provider}")
