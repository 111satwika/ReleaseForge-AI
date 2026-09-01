"""Unit tests for app/llm/cost.py — pure arithmetic, no I/O, no real provider needed."""

from app.llm.cost import estimate_cost, is_known_paid_model


def test_unknown_model_is_not_known_paid():
    assert is_known_paid_model("qwen2.5:7b-instruct") is False
    assert is_known_paid_model("gpt-4o") is True


def test_estimate_cost_is_none_for_an_unrecognized_model():
    """An unknown model (every local Ollama model, today) must be None, not a fake 0.0 —
    the same None-vs-zero discipline eval/metrics.py already holds everywhere else. A
    local/free model isn't "$0.00 of cost," it's "not priced in this table.\""""
    result = estimate_cost("qwen2.5:7b-instruct", prompt_tokens=1000, completion_tokens=500)
    assert result.estimated_cost_usd is None
    assert result.prompt_tokens == 1000
    assert result.completion_tokens == 500


def test_estimate_cost_computes_real_dollar_amount_for_a_known_model():
    # gpt-4o-mini: $0.00015 / 1K input, $0.0006 / 1K output
    result = estimate_cost("gpt-4o-mini", prompt_tokens=2000, completion_tokens=1000)
    expected = (2000 / 1000) * 0.00015 + (1000 / 1000) * 0.0006
    assert result.estimated_cost_usd == round(expected, 6)


def test_estimate_cost_zero_tokens_is_zero_dollars_not_none():
    """Zero tokens against a known model is a real, computable $0.00 — different from an
    unrecognized model, which is genuinely unknown, not zero."""
    result = estimate_cost("gpt-4o", prompt_tokens=0, completion_tokens=0)
    assert result.estimated_cost_usd == 0.0


def test_embedding_model_cost_ignores_completion_tokens():
    """Embeddings have no output side — OpenAIProvider.embed() always passes
    completion_tokens=0, and the pricing table's output rate for embedding models is 0.0
    to match, so passing a nonzero completion_tokens here still can't inflate the cost."""
    result = estimate_cost("text-embedding-3-small", prompt_tokens=1000, completion_tokens=999)
    assert result.estimated_cost_usd == round(1.0 * 0.00002, 6)
