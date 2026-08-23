from openai import AsyncOpenAI

from app.observability.tracing import get_tracer, record_exception

from .base import LLMMessage, LLMResponse


class OllamaProvider:
    """Talks to a local Ollama daemon via its OpenAI-compatible endpoint.

    Because the wire format matches OpenAI's, swapping to real OpenAI later
    is just a different base_url/api_key/model — no adapter rewrite needed.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        embedding_model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        tracer = get_tracer()
        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.message_count", len(messages))
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )
            except Exception as exc:
                record_exception(span, exc)
                raise

            content = response.choices[0].message.content or ""
            span.set_attribute("llm.response_chars", len(content))
            # Ollama's OpenAI-compatible endpoint does return real usage stats (not every
            # OpenAI-compatible server does) — record them when present rather than
            # assuming they'll always be there.
            if response.usage is not None:
                span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
                span.set_attribute("llm.total_tokens", response.usage.total_tokens)
            return LLMResponse(content=content, model=self.model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Uses a separate, smaller embedding model (`nomic-embed-text` by default) —
        not the chat model — since embeddings are a distinct Ollama pull or Ollama returns
        a 404 for a chat-only model asked to embed. Caller must have pulled it once:
        `ollama pull nomic-embed-text`."""
        tracer = get_tracer()
        with tracer.start_as_current_span("llm.embed") as span:
            span.set_attribute("llm.embedding_model", self.embedding_model)
            span.set_attribute("llm.embed_text_count", len(texts))
            try:
                response = await self._client.embeddings.create(model=self.embedding_model, input=texts)
            except Exception as exc:
                record_exception(span, exc)
                raise
            if response.usage is not None:
                span.set_attribute("llm.total_tokens", response.usage.total_tokens)
            return [item.embedding for item in response.data]
