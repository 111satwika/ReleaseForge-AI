"""Unit tests for app/eval/rag_eval.py's pure metric functions — no embedding model or
retriever involved, just hand-built ranked id lists. judge_faithfulness() is exercised
with a fake LLM (canned chat responses) so the JSON-parsing/fallback logic is verified
without needing a real judge model — the same "prove the logic, not the model" split
Phase 6's own tests already use for embeddings."""

import pytest

from app.eval.rag_eval import (
    EvalQuery,
    judge_faithfulness,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval_results,
)
from app.llm.base import LLMResponse


def test_precision_at_k_counts_relevant_hits_in_top_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 2 / 3


def test_precision_at_k_only_looks_at_top_k_not_the_whole_list():
    assert precision_at_k(["a", "b", "c", "d"], {"d"}, k=2) == 0.0


def test_precision_at_k_empty_retrieval_is_zero_not_a_crash():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_recall_at_k_divides_by_total_relevant_not_by_k():
    # 3 relevant ids exist total, only 1 was retrieved in the top 2 -> 1/3
    assert recall_at_k(["a", "x"], {"a", "b", "c"}, k=2) == 1 / 3


def test_recall_at_k_with_no_relevant_ids_is_zero_not_division_by_zero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_reciprocal_rank_finds_first_relevant_position():
    assert reciprocal_rank(["x", "y", "a", "z"], {"a"}) == 1 / 3


def test_reciprocal_rank_is_zero_when_nothing_relevant_retrieved():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_perfect_ranking_scores_1():
    # both relevant ids appear first, in any order -> ideal ordering achieved
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_ndcg_worse_ranking_scores_less_than_perfect():
    perfect = ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=3)
    worse = ndcg_at_k(["x", "a", "b"], {"a", "b"}, k=3)
    assert worse < perfect


def test_ndcg_no_relevant_docs_at_all_is_zero():
    assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


def test_score_retrieval_results_aggregates_across_queries():
    queries = [
        EvalQuery(query="q1", relevant_ids={"a"}),
        EvalQuery(query="q2", relevant_ids={"z"}),
    ]
    retrieved = [
        ["a", "b", "c"],  # perfect hit at rank 1
        ["x", "y", "w"],  # complete miss
    ]
    report = score_retrieval_results(queries, retrieved, k=3)
    assert report.queries_evaluated == 2
    assert report.mean_reciprocal_rank == 0.5  # (1.0 + 0.0) / 2
    assert report.per_query[0]["reciprocal_rank"] == 1.0
    assert report.per_query[1]["reciprocal_rank"] == 0.0


# ---------- faithfulness (LLM-as-judge, fake LLM so no real model is needed) ----------


class _FakeJudgeLLM:
    """Returns exactly one canned response, whatever it's asked — enough to exercise
    judge_faithfulness()'s parsing without a real model."""

    def __init__(self, content: str):
        self._content = content

    async def chat(self, messages):
        return LLMResponse(content=self._content, model="fake-judge")


@pytest.mark.asyncio
async def test_judge_faithfulness_true_when_judge_says_so():
    llm = _FakeJudgeLLM('{"faithful": true, "unsupported_claims": [], "explanation": "Matches the passage."}')
    result = await judge_faithfulness(llm, "q", "a", ["some passage"])
    assert result.faithful is True
    assert result.unsupported_claims == []


@pytest.mark.asyncio
async def test_judge_faithfulness_flags_unsupported_claims():
    # The exact real failure this was built to catch: the passage says Chroma replaced
    # Qdrant, but the checked answer claims Qdrant — a real live-observed case, not a
    # hypothetical one (see the Phase 6 build-plan addendum).
    llm = _FakeJudgeLLM(
        '{"faithful": false, "unsupported_claims": ["dense retrieval uses Qdrant"], '
        '"explanation": "The passage says Chroma replaced Qdrant, not that Qdrant is used."}'
    )
    passages = ["backend/app/rag/vector_store.py wraps Chroma... instead of Qdrant"]
    result = await judge_faithfulness(llm, "what vector db is used?", "It uses Qdrant.", passages)
    assert result.faithful is False
    assert "Qdrant" in result.unsupported_claims[0]
    assert result.explanation


@pytest.mark.asyncio
async def test_judge_faithfulness_extracts_json_from_a_fenced_code_block():
    llm = _FakeJudgeLLM('Here you go:\n```json\n{"faithful": true, "unsupported_claims": [], "explanation": "ok"}\n```')
    result = await judge_faithfulness(llm, "q", "a", ["p"])
    assert result.faithful is True


@pytest.mark.asyncio
async def test_judge_faithfulness_treats_malformed_json_as_not_faithful():
    # A judge that fails to return usable JSON must not silently pass the answer through
    # — that would be worse than having no check at all.
    llm = _FakeJudgeLLM("I think it looks fine, no JSON here.")
    result = await judge_faithfulness(llm, "q", "a", ["p"])
    assert result.faithful is False
    assert "valid JSON" in result.explanation
