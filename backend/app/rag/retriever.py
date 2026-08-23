from typing import Awaitable, Callable

from app.rag.keyword_store import KeywordStore
from app.rag.vector_store import VectorStore

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

RRF_K = 60  # standard reciprocal-rank-fusion constant; not sensitive to tuning at this corpus size


async def hybrid_retrieve(
    query: str,
    embed_fn: EmbedFn,
    vector_store: VectorStore,
    k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Dense (embedding similarity) + sparse (BM25 keyword) retrieval, merged by
    reciprocal rank fusion rather than picking one. Dense catches paraphrases ("what
    broke in staging" ~ "deployment failure"); sparse catches exact identifiers dense
    embeddings often blur (a file path, a branch name, an error string). Fusing ranks
    instead of raw scores sidesteps the two methods using incomparable score scales.

    Callers must not let this raise into a user-facing failure — the embedding call
    depends on Ollama actually running with the embedding model pulled, which is not
    guaranteed. Wrap the call site in a try/except and degrade to no retrieved context."""
    [query_embedding] = await embed_fn([query])
    dense_hits = vector_store.query(query_embedding, k=k * 2, where=where)

    corpus = vector_store.get_all(where=where)
    sparse_hits = KeywordStore(corpus).query(query, k=k * 2)

    fused: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for rank, hit in enumerate(dense_hits):
        fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id[hit["id"]] = hit
    for rank, hit in enumerate(sparse_hits):
        fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id.setdefault(hit["id"], hit)

    ranked_ids = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
    return [{**by_id[i], "fused_score": fused[i]} for i in ranked_ids]
