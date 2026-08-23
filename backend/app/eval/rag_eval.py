"""RAG retrieval-quality metrics — design doc Section 16 "RAG": Recall, Precision, MRR,
NDCG. Faithfulness isn't included: that needs an LLM judging whether a *generated
answer* is actually supported by the retrieved text, a different kind of evaluation
(answer quality, not retrieval quality) this module doesn't attempt.

Every metric function here is pure — ranked result IDs in, a number out — and unit
tested without any embedding model. `evaluate_retrieval()` is the one function that
needs a real, running retriever (hybrid_retrieve against a real vector store with a real
embed_fn), the same Ollama-shaped dependency Phase 6's ingestion already has. This
module was built and unit-tested in an environment with no Ollama running — see the
Phase 8 build-plan entry for exactly what that does and doesn't prove.
"""

import math
from dataclasses import dataclass


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
