"""Unit tests for app/eval/rag_eval.py's pure metric functions — no embedding model or
retriever involved, just hand-built ranked id lists."""

import math

from app.eval.rag_eval import (
    EvalQuery,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval_results,
)


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
