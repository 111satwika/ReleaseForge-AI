import asyncio
import hashlib
from pathlib import Path

from app.llm.base import LLMProvider
from app.rag.chunking import Chunk, chunk_markdown, chunk_release_record
from app.rag.graph_store import GraphStore
from app.rag.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOC_FILES = [
    REPO_ROOT / "ReleaseForge_AI_Complete_Project_Design.md",
    REPO_ROOT / "RELEASEFORGE_BUILD_PLAN.md",
]
DOCS_HASH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "docs_hash.txt"


async def _embed_and_store(llm: LLMProvider, vector_store: VectorStore, chunks: list[Chunk]) -> None:
    if not chunks:
        return
    embeddings = await llm.embed([c.text for c in chunks])
    vector_store.add(chunks, embeddings)


async def ingest_project_docs(llm: LLMProvider, vector_store: VectorStore) -> int:
    """Indexes ReleaseForge's own design/build-plan docs as the "engineering
    documentation" corpus from the design doc's Section 7 — the RAG concept needs some
    real documents to retrieve over, and these are the ones that actually exist and
    actually change as the project evolves.

    Idempotent by content hash: an unchanged doc on the next backend restart costs zero
    embedding calls, not a silent full re-embed every startup."""
    texts = [f.read_text(encoding="utf-8") for f in DOC_FILES if f.exists()]
    if not texts:
        return 0

    combined_hash = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
    if DOCS_HASH_PATH.exists() and DOCS_HASH_PATH.read_text().strip() == combined_hash:
        return 0

    chunks: list[Chunk] = []
    for f, text in zip((f for f in DOC_FILES if f.exists()), texts):
        chunks.extend(chunk_markdown(text, source=f.name))

    await _embed_and_store(llm, vector_store, chunks)
    DOCS_HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_HASH_PATH.write_text(combined_hash)
    return len(chunks)


async def ingest_release_record(
    llm: LLMProvider, vector_store: VectorStore, graph_store: GraphStore, record
) -> None:
    """Called right after a task run is recorded to release memory (main.py's
    `_maybe_record_release`) — every new release event becomes retrievable and joins
    the relationship graph immediately, not on some later batch job."""
    chunk = chunk_release_record(record)
    await _embed_and_store(llm, vector_store, [chunk])
    await asyncio.to_thread(graph_store.upsert_release, record)
