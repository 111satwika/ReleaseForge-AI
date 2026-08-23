"""Tests for app/rag/reranker.py — the LLM-prompted "cross-encoder reranking" substitute
(see the module's own docstring for why it's LLM-prompted, not a literal cross-encoder).
A fake chat function stands in for OllamaProvider.chat(), same "prove the logic without
a real model" approach as the rest of this test suite."""

import pytest

from app.llm.base import LLMResponse
from app.rag.chunking import Chunk
from app.rag.reranker import hybrid_retrieve_reranked, rerank
from app.rag.vector_store import VectorStore


def _hit(id_, text):
    return {"id": id_, "text": text, "metadata": {}, "distance": 0.1}


async def _chat_returning(content: str):
    async def chat_fn(messages):
        return LLMResponse(content=content, model="fake-reranker")

    return chat_fn


@pytest.mark.asyncio
async def test_rerank_is_a_noop_when_there_are_not_more_hits_than_top_k():
    hits = [_hit("a", "x"), _hit("b", "y")]
    chat_fn = await _chat_returning("should never be called")
    result = await rerank(chat_fn, "query", hits, top_k=4)
    assert result == hits


@pytest.mark.asyncio
async def test_rerank_reorders_hits_per_the_llms_ranking():
    hits = [_hit("a", "irrelevant"), _hit("b", "somewhat relevant"), _hit("c", "very relevant"), _hit("d", "filler")]
    chat_fn = await _chat_returning("[2, 1]")  # c, then b
    result = await rerank(chat_fn, "query", hits, top_k=2)
    assert [h["id"] for h in result] == ["c", "b"]


@pytest.mark.asyncio
async def test_rerank_extracts_json_from_a_fenced_code_block():
    hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
    chat_fn = await _chat_returning("Sure, here you go:\n```json\n[1, 0]\n```")
    result = await rerank(chat_fn, "query", hits, top_k=2)
    assert [h["id"] for h in result] == ["b", "a"]


@pytest.mark.asyncio
async def test_rerank_fills_remaining_slots_when_the_llm_returns_too_few_indices():
    hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
    chat_fn = await _chat_returning("[1]")  # only one index for a top_k of 2
    result = await rerank(chat_fn, "query", hits, top_k=2)
    assert result[0]["id"] == "b"
    assert result[1]["id"] in ("a", "c")  # filled from original RRF order, not fabricated


@pytest.mark.asyncio
async def test_rerank_fails_open_to_original_order_on_unparseable_output():
    hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]
    chat_fn = await _chat_returning("I don't understand the question, sorry!")
    result = await rerank(chat_fn, "query", hits, top_k=2)
    assert [h["id"] for h in result] == ["a", "b"]  # original order, first top_k


@pytest.mark.asyncio
async def test_rerank_fails_open_when_the_chat_call_itself_raises():
    hits = [_hit("a", "x"), _hit("b", "y"), _hit("c", "z")]

    async def broken_chat_fn(messages):
        raise RuntimeError("Ollama isn't running")

    result = await rerank(broken_chat_fn, "query", hits, top_k=2)
    assert [h["id"] for h in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_hybrid_retrieve_reranked_narrows_a_wider_pool_down_to_k(tmp_path):
    async def fake_embed(texts):
        return [[float(len(t)), float(t.count("relevant"))] for t in texts]

    store = VectorStore(persist_dir=tmp_path / "chroma")
    chunks = [
        Chunk(id="a", text="deployment strategy notes, not very relevant to auth", source="test"),
        Chunk(id="b", text="authentication and login relevant details here", source="test"),
        Chunk(id="c", text="unrelated changelog entry", source="test"),
    ]
    embeddings = await fake_embed([c.text for c in chunks])
    store.add(chunks, embeddings)

    async def chat_fn(messages):
        # Find whichever index the prompt actually assigned the "login" passage —
        # RRF fusion order among a/b/c isn't something this test should assume.
        prompt = messages[0].content
        for line in prompt.split("\n"):
            if line.startswith("[") and "login" in line:
                index = line[1 : line.index("]")]
                return LLMResponse(content=f"[{index}]", model="fake-reranker")
        raise AssertionError("expected passage not found in the reranking prompt")

    hits = await hybrid_retrieve_reranked("login relevant", fake_embed, chat_fn, store, k=1, fetch_k=3)
    assert len(hits) == 1
    assert hits[0]["id"] == "b"
