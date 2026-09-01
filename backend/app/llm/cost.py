"""Cost estimation for LLM calls — the design doc's Section 16 "cost tracking" item,
left unbuilt through Phase 8 because Ollama (this project's only wired-in provider) is
free/local, so there was no real $ signal to compute against; Phase 8's own build-plan
entry already named this exact plan ("the token-usage attributes already captured are
what a future OpenAI/Anthropic billing calculation would multiply against").

Built ahead of a real paid provider actually existing — llm/factory.py still only ever
returns OllamaProvider; the "openai / anthropic providers get added here later" branch
there is still unbuilt, a separate and considerably bigger piece of work than pricing
math. This module is genuinely usable the moment that branch exists, not before.

Pricing is real, publicly published, checked directly against each vendor's own pricing
page (not guessed) as of 2026-08. It's a snapshot, not a live-fetched source of truth —
prices change, and this table should be revisited whenever a real paid provider is
actually wired in."""

from dataclasses import dataclass

# model name -> (input $ per 1K tokens, output $ per 1K tokens), USD.
PRICING_PER_1K_TOKENS_USD: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-nano": (0.0001, 0.0004),
    "o1": (0.015, 0.060),
    "o1-mini": (0.0011, 0.0044),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-haiku-20241022": (0.0008, 0.004),
    "claude-3-opus-20240229": (0.015, 0.075),
    # Embedding models have no "output" side — input rate only, output rate 0.0.
    # embed() below always passes completion_tokens=0, so this never actually multiplies.
    "text-embedding-3-small": (0.00002, 0.0),
    "text-embedding-3-large": (0.00013, 0.0),
}


@dataclass
class CostEstimate:
    model: str
    prompt_tokens: int
    completion_tokens: int
    # None, honestly, not 0.0 — this project has avoided "unknown treated as zero"
    # everywhere else (see eval/metrics.py's whole None-vs-0.0 discipline), and cost is
    # no different: a local/free/unrecognized model isn't "free," it's "not priced here."
    estimated_cost_usd: float | None


def is_known_paid_model(model: str) -> bool:
    return model in PRICING_PER_1K_TOKENS_USD


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> CostEstimate:
    pricing = PRICING_PER_1K_TOKENS_USD.get(model)
    if pricing is None:
        return CostEstimate(
            model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, estimated_cost_usd=None
        )
    input_rate, output_rate = pricing
    cost = (prompt_tokens / 1000) * input_rate + (completion_tokens / 1000) * output_rate
    return CostEstimate(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, estimated_cost_usd=round(cost, 6)
    )
