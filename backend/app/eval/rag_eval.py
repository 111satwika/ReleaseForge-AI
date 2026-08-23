"""RAG retrieval-quality metrics — design doc Section 16 "RAG": Recall, Precision, MRR,
NDCG — plus, as of the addendum below, faithfulness: whether a generated *answer* is
actually supported by the retrieved text, a different kind of evaluation (answer
quality, not retrieval quality) than the rank-based metrics above measure.

The retrieval metric functions here are pure — ranked result IDs in, a number out — and
unit tested without any embedding model. `evaluate_retrieval()` is the one function that
needs a real, running retriever (hybrid_retrieve against a real vector store with a real
embed_fn), the same Ollama-shaped dependency Phase 6's ingestion already has. This
module was originally built and unit-tested in an environment with no Ollama running —
see the Phase 8 build-plan entry for exactly what that did and didn't prove at the time.
"""

import json
import math
import re
from dataclasses import dataclass, field

from app.llm.base import LLMMessage


@dataclass
class EvalQuery:
    """One evaluation example: a real question, and the chunk id(s) that actually answer
    it. Building a genuine eval set means knowing the corpus well enough to say "chunk X
    is the right answer to this" — these should come from someone who actually knows
    the indexed content, not be invented generically."""

    query: str
    relevant_ids: set[str]


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant result, or 0 if none of the retrieved ids are
    relevant at all — the per-query building block of Mean Reciprocal Rank."""
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance NDCG@k: every relevant id is worth the same (this codebase has
    no graded relevance labels — a chunk is either a real answer or it isn't), so this
    reduces to comparing the retrieved ranking's discounted gain against the best
    possible ordering (all relevant ids first)."""
    top_k = retrieved_ids[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item_id in enumerate(top_k, start=1) if item_id in relevant_ids)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalEvalReport:
    queries_evaluated: int
    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_ndcg_at_k: float
    per_query: list[dict]


def score_retrieval_results(eval_queries: list[EvalQuery], retrieved_per_query: list[list[str]], k: int = 5) -> RetrievalEvalReport:
    """The pure half: takes already-retrieved ranked id lists (however they were
    produced) and scores them. Separated from evaluate_retrieval() below so the scoring
    logic itself is testable with hand-built rankings, no retriever or embedding model
    required."""
    per_query = []
    for eq, retrieved_ids in zip(eval_queries, retrieved_per_query):
        per_query.append(
            {
                "query": eq.query,
                "precision_at_k": precision_at_k(retrieved_ids, eq.relevant_ids, k),
                "recall_at_k": recall_at_k(retrieved_ids, eq.relevant_ids, k),
                "reciprocal_rank": reciprocal_rank(retrieved_ids, eq.relevant_ids),
                "ndcg_at_k": ndcg_at_k(retrieved_ids, eq.relevant_ids, k),
            }
        )
    n = len(per_query) or 1
    return RetrievalEvalReport(
        queries_evaluated=len(per_query),
        k=k,
        mean_precision_at_k=round(sum(q["precision_at_k"] for q in per_query) / n, 4),
        mean_recall_at_k=round(sum(q["recall_at_k"] for q in per_query) / n, 4),
        mean_reciprocal_rank=round(sum(q["reciprocal_rank"] for q in per_query) / n, 4),
        mean_ndcg_at_k=round(sum(q["ndcg_at_k"] for q in per_query) / n, 4),
        per_query=per_query,
    )


async def evaluate_retrieval(eval_queries: list[EvalQuery], embed_fn, vector_store, k: int = 5) -> RetrievalEvalReport:
    """The I/O half — runs the real hybrid retriever (Phase 6) for each eval query and
    scores the results. Needs a reachable embedding model; raises the same way
    hybrid_retrieve itself does if Ollama isn't running, rather than masking that."""
    from app.rag.retriever import hybrid_retrieve

    retrieved_per_query = []
    for eq in eval_queries:
        hits = await hybrid_retrieve(eq.query, embed_fn, vector_store, k=k)
        retrieved_per_query.append([h["id"] for h in hits])
    return score_retrieval_results(eval_queries, retrieved_per_query, k=k)


def _extract_json(text: str) -> dict | None:
    """Same tolerant extraction graph.py's classify_intent/review nodes already rely
    on for LLM-as-judge output — a fenced ```json block if present, otherwise the first
    {...} span. Kept as a small local copy rather than importing from app.agents.graph:
    this module has no other dependency on the agent graph, and one shouldn't be added
    just to share thirteen lines."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


@dataclass
class FaithfulnessResult:
    """faithful=False with an empty unsupported_claims list means the judge itself
    failed to return usable JSON — treated as a failure to verify, not a pass, since a
    faithfulness check that silently no-ops on a malformed judge response would be
    worse than not having one."""

    faithful: bool
    unsupported_claims: list[str] = field(default_factory=list)
    explanation: str = ""


async def judge_faithfulness(llm, question: str, answer: str, passages: list[str]) -> FaithfulnessResult:
    """LLM-as-judge: does `answer` only make claims the retrieved `passages` actually
    support? This is the answer-quality half rag_eval.py originally didn't attempt — the
    module docstring used to say so outright. Built after a real, live-observed failure
    of exactly this kind (see the Phase 6 build-plan addendum): a grounded chat answer
    correctly cited real retrieved text but still misstated one fact from it (reporting
    "Qdrant" when the retrieved passage said Chroma replaced Qdrant). Retrieval was
    correct; faithfulness wasn't — two different things, which is exactly why this is a
    separate check from precision/recall/MRR/NDCG above, not folded into them."""
    passages_block = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages)) or "(no passages retrieved)"
    prompt = (
        "You are a strict fact-checker for a RAG system. Given source passages and an "
        "answer that was supposed to be grounded in them, find any claim in the answer "
        "that the passages do not actually support — including a claim that contradicts "
        "what a passage says, not just one the passages never mention at all.\n\n"
        f"Source passages:\n{passages_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer to check:\n{answer}\n\n"
        "Reply with exactly one JSON object, no other text:\n"
        '{"faithful": true or false, "unsupported_claims": ["...", ...], '
        '"explanation": "one sentence"}'
    )
    response = await llm.chat([LLMMessage(role="user", content=prompt)])
    parsed = _extract_json(response.content)
    if parsed is None:
        return FaithfulnessResult(
            faithful=False,
            unsupported_claims=[],
            explanation=f"Judge response wasn't valid JSON: {response.content[:200]!r}",
        )
    return FaithfulnessResult(
        faithful=bool(parsed.get("faithful", False)),
        unsupported_claims=[str(c) for c in (parsed.get("unsupported_claims") or [])],
        explanation=str(parsed.get("explanation", "")),
    )
