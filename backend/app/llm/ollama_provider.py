from openai import AsyncOpenAI

from .base import LLMMessage, LLMResponse


class OllamaProvider:
    """Talks to a local Ollama daemon via its OpenAI-compatible endpoint.

    Because the wire format matches OpenAI's, swapping to real OpenAI later
    is just a different base_url/api_key/model — no adapter rewrite needed.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )
        return LLMResponse(content=response.choices[0].message.content or "", model=self.model)
