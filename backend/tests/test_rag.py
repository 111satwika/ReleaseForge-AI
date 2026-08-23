"""RAG module tests. These deliberately avoid needing a real Ollama daemon: the vector
store is exercised with hand-built (fake) embeddings, injected the same way the real
OllamaProvider.embed() would be, so the tests prove the retrieval/graph logic itself is
correct independent of whether an embedding model is actually installed."""

from datetime import datetime, timezone

import pytest

from app.rag.chunking import Chunk, chunk_markdown, chunk_release_record
from app.rag.graph_store import GraphStore
from app.rag.keyword_store import KeywordStore
from app.rag.retriever import hybrid_retrieve
from app.rag.vector_store import VectorStore

# Each test gets pytest's own unique `tmp_path`, not a shared directory wiped between
# tests — Chroma's PersistentClient keeps its sqlite file open for the process's life,
# and Windows (unlike POSIX) refuses to delete a file that's still open elsewhere, so a
# shared "clean up before every test" fixture fails on the second test to touch Chroma.


def _fake_embed(dims: int = 8):
    """A deterministic, content-sensitive "embedding": counts a few keyword buckets so
    semantically similar fake texts land close together, without any real model."""
    buckets = ["deploy", "test", "review", "docker", "auth", "login", "readiness", "graph"]

    async def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(b)) for b in buckets[:dims]])
        return vectors

    return embed


# ---------- chunking ----------


def test_chunk_markdown_splits_on_headings():
    text = "# Title\nintro text\n\n## Section A\nbody A\n\n## Section B\nbody B\n"
    chunks = chunk_markdown(text, source="doc.md")
    headings = [c.metadata["heading"] for c in chunks]
    assert "Section A" in headings
    assert "Section B" in headings
    assert all(c.source == "doc.md" for c in chunks)


def test_chunk_markdown_is_content_addressed_and_idempotent():
    text = "## Same heading\nsame body\n"
    a = chunk_markdown(text, source="doc.md")
    b = chunk_markdown(text, source="doc.md")
    assert [c.id for c in a] == [c.id for c in b]


def test_chunk_markdown_splits_long_sections_on_paragraph_budget():
    long_body = "\n\n".join(f"Paragraph {i} " + "x" * 200 for i in range(10))
    text = f"## Big section\n{long_body}\n"
    chunks = chunk_markdown(text, source="doc.md", max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)  # budget plus one paragraph's slack


class _FakeRecord:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", 1)
        self.workspace_path = kwargs.get("workspace_path", "/repo")
        self.outcome = kwargs.get("outcome", "docker_built")
        self.target_file = kwargs.get("target_file", "app.py")
        self.change_description = kwargs.get("change_description", "add rate limiting")
        self.test_passed = kwargs.get("test_passed", True)
        self.review_status = kwargs.get("review_status", "APPROVED")
        self.review_risk = kwargs.get("review_risk", "Low")
        self.readiness_score = kwargs.get("readiness_score", 100)
        self.readiness_recommendation = kwargs.get("readiness_recommendation", "GO")
        self.pr_url = kwargs.get("pr_url", "https://github.com/x/y/pull/1")
        self.created_at = kwargs.get("created_at", datetime(2026, 8, 21, tzinfo=timezone.utc))


def test_chunk_release_record_contains_key_facts():
    record = _FakeRecord()
    chunk = chunk_release_record(record)
    assert "app.py" in chunk.text
    assert "GO" in chunk.text
    assert chunk.metadata["outcome"] == "docker_built"
    assert chunk.source == "release-history"


# ---------- vector + keyword + hybrid retrieval ----------


@pytest.mark.asyncio
async def test_hybrid_retrieve_finds_relevant_chunk_over_irrelevant_ones(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    embed = _fake_embed()

    docs_text = {
        "a": "We added Docker build support and a Dockerfile generator.",
        "b": "Login page authentication was reviewed and approved.",
        "c": "Unrelated paragraph about release readiness scoring.",
    }
    chunks = [Chunk(id=k, text=v, source="test") for k, v in docs_text.items()]
    embeddings = await embed([c.text for c in chunks])
    store.add(chunks, embeddings)

    hits = await hybrid_retrieve("tell me about the docker build", embed, store, k=2)
    assert hits
    assert hits[0]["id"] == "a"


@pytest.mark.asyncio
async def test_hybrid_retrieve_degrades_to_empty_when_store_is_empty(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma_empty")
    hits = await hybrid_retrieve("anything", _fake_embed(), store, k=3)
    assert hits == []


def test_keyword_store_ranks_exact_term_match_first():
    docs = [
        {"id": "1", "text": "the quick brown fox"},
        {"id": "2", "text": "release readiness score was 94"},
        {"id": "3", "text": "nothing relevant here at all"},
    ]
    store = KeywordStore(docs)
    hits = store.query("readiness score", k=2)
    assert hits[0]["id"] == "2"


# ---------- graph store ----------


def test_graph_store_upsert_and_query_relationships(tmp_path):
    graph = GraphStore(db_path=tmp_path / "graph.db")
    record = _FakeRecord(id=1, target_file="login.py")
    graph.upsert_release(record)

    touching = graph.releases_touching_file("/repo", "login.py")
    assert len(touching) == 1

    facts = graph.recent_relationships("/repo")
    relations = {f["relation"] for f in facts}
    assert "HAS_RELEASE" in relations
    assert "MODIFIED" in relations


def test_graph_store_is_idempotent_on_same_release(tmp_path):
    graph = GraphStore(db_path=tmp_path / "graph_idempotent.db")
    record = _FakeRecord(id=7, target_file="app.py")
    graph.upsert_release(record)
    graph.upsert_release(record)  # simulate a duplicate call

    facts = graph.recent_relationships("/repo", limit=100)
    modified_edges = [f for f in facts if f["relation"] == "MODIFIED"]
    assert len(modified_edges) == 1
