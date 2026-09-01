import os

from openai import AsyncOpenAI

from app.llm.cost import estimate_cost
from app.observability.tracing import get_tracer, record_exception

from .base import LLMMessage, LLMResponse


class OpenAIProvider:
    """Talks to the real OpenAI API. Structurally mirrors OllamaProvider (same
    AsyncOpenAI client, same chat()/embed() shape) rather than sharing a base class with
    it — the two are genuinely identical today, but OllamaProvider is already covered by
    tests elsewhere in this suite, and duplicating ~70 lines here is a smaller risk than
    refactoring already-tested code for the sake of removing that duplication.

    API key is read from OPENAI_API_KEY (the standard env var the openai SDK itself
    expects) — never hardcoded, never logged, never passed on the command line.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Set it as a real environment variable "
                "(not hardcoded in code) before selecting RELEASEFORGE_LLM_PROVIDER=openai."
            )
        self._client = AsyncOpenAI(api_key=resolved_key)

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
            if response.usage is not None:
                span.set_attribute("llm.prompt_tokens", response.usage.prompt_tokens)
                span.set_attribute("llm.completion_tokens", response.usage.completion_tokens)
                span.set_attribute("llm.total_tokens", response.usage.total_tokens)
                # Unlike Ollama, this model IS in the paid pricing table — a real,
                # non-None cost gets recorded on every call, not just wired ahead of time.
                cost = estimate_cost(self.model, response.usage.prompt_tokens, response.usage.completion_tokens)
                if cost.estimated_cost_usd is not None:
                    span.set_attribute("llm.estimated_cost_usd", cost.estimated_cost_usd)
            return LLMResponse(content=content, model=self.model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
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
                cost = estimate_cost(self.embedding_model, response.usage.total_tokens, 0)
                if cost.estimated_cost_usd is not None:
                    span.set_attribute("llm.estimated_cost_usd", cost.estimated_cost_usd)
            return [item.embedding for item in response.data]
