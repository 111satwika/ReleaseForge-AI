"""Second-pass reranking over hybrid_retrieve()'s fused results — closes the "cross-encoder
reranking" line item named (and left "not yet") in the build plan's concept-coverage table.

Named honestly rather than mislabeled: a literal cross-encoder (a sentence-transformers-
style classifier head, e.g. bge-reranker or ms-marco-MiniLM) isn't available anywhere in
this Ollama-only stack — checked directly against Ollama's own model registry, nothing
of that model family is hosted there, and Ollama's chat/embed API surface has no
cross-encoder-shaped endpoint to begin with. What's built here is LLM-prompted
reranking: the chat model scores each candidate passage's relevance to the query
directly. This is a real, commonly-used technique that does the same second-pass
quality-filtering job a cross-encoder would — but it is a genuinely different mechanism,
not a cross-encoder running locally, and calling it one would be exactly the kind of
overclaim this project has avoided everywhere else.
"""

import json
import re

from app.llm.base import LLMMessage
from app.rag.retriever import EmbedFn, hybrid_retrieve
from app.rag.vector_store import VectorStore


def _extract_json_list(text: str) -> list | None:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brackets = re.search(r"\[.*\]", text, re.DOTALL)
        if brackets:
            text = brackets.group(0)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


async def rerank(chat_fn, query: str, hits: list[dict], top_k: int) -> list[dict]:
    """chat_fn: an LLMProvider.chat-shaped callable (async, takes list[LLMMessage],
    returns something with .content) — passed explicitly rather than a whole provider so
    this stays testable with a bare async function, no fake-provider object needed.

    Fails open, not closed: if the LLM call errors or returns unparseable output, this
    returns hits[:top_k] in their original (RRF) order rather than raising — a reranking
    step that can take down retrieval entirely would be a worse trade than just skipping
    it, matching the "never crash a chat turn over an enhancement" principle Phase 6
    already established for retrieval itself."""
    if not hits:
        return []
    if len(hits) <= top_k:
        return hits

    passages_block = "\n\n".join(f"[{i}] {h['text'][:600]}" for i, h in enumerate(hits))
    prompt = (
        "Rank the following passages by how relevant they are to the query, most "
        f"relevant first. Return exactly the top {top_k} as a JSON array of their "
        "index numbers only, e.g. [3, 0, 4] — no other text.\n\n"
        f"Query: {query}\n\nPassages:\n{passages_block}"
    )
    try:
        response = await chat_fn([LLMMessage(role="user", content=prompt)])
        indices = _extract_json_list(response.content)
        if indices is None:
            return hits[:top_k]
        reranked = [hits[int(i)] for i in indices if isinstance(i, int) and 0 <= i < len(hits)]
        # A malformed or partial ranking (an out-of-range index, a duplicate, fewer than
        # top_k parsed) still degrades to *something* useful rather than nothing: fill
        # any remaining slots from the original RRF order, skipping what's already in.
        if len(reranked) < top_k:
            seen_ids = {h["id"] for h in reranked}
            for h in hits:
                if len(reranked) >= top_k:
                    break
                if h["id"] not in seen_ids:
                    reranked.append(h)
                    seen_ids.add(h["id"])
        return reranked[:top_k]
    except Exception:
        return hits[:top_k]


async def hybrid_retrieve_reranked(
    query: str,
    embed_fn: EmbedFn,
    chat_fn,
    vector_store: VectorStore,
    k: int = 4,
    fetch_k: int = 10,
    where: dict | None = None,
) -> list[dict]:
    """Fetches a wider candidate pool via hybrid_retrieve() (dense+sparse fusion), then
    narrows it back down to k with the LLM reranking pass above. fetch_k > k on purpose —
    reranking only adds value with more candidates to actually choose among than the
    final k; reranking an already-k-sized list is a no-op."""
    candidates = await hybrid_retrieve(query, embed_fn, vector_store, k=fetch_k, where=where)
    return await rerank(chat_fn, query, candidates, top_k=k)
