from typing import Protocol

from pydantic import BaseModel


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str


class LLMProvider(Protocol):
    """Every model backend (Ollama, OpenAI, Anthropic, ...) implements this."""

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
