"""Tests for app/rag/ingest.py's embed-then-write glue — previously untested (only the
pure chunk_*() functions and hybrid_retrieve() had coverage). Same "prove the logic
without a real model" approach as test_rag.py: a deterministic fake embed function
stands in for OllamaProvider.embed()."""

from datetime import datetime, timezone

import pytest

from app.rag.graph_store import GraphStore
from app.rag.ingest import ingest_incident_record, ingest_project_docs, ingest_release_record
from app.rag.vector_store import VectorStore


class _FakeLLM:
    """embed() only — ingest.py never calls chat()."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(t.count(" "))] for t in texts]


class _FakeRelease:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", 1)
        self.workspace_path = kwargs.get("workspace_path", "/repo")
        self.outcome = kwargs.get("outcome", "rolled_back")
        self.target_file = kwargs.get("target_file", "app.py")
        self.change_description = kwargs.get("change_description", "batch the writer")
        self.test_passed = kwargs.get("test_passed", True)
        self.review_status = kwargs.get("review_status", "APPROVED")
        self.review_risk = kwargs.get("review_risk", "Low")
        self.readiness_score = kwargs.get("readiness_score", 90)
        self.readiness_recommendation = kwargs.get("readiness_recommendation", "GO")
        self.pr_url = kwargs.get("pr_url", "https://github.com/x/y/pull/9")
        self.created_at = kwargs.get("created_at", datetime(2026, 8, 21, tzinfo=timezone.utc))


class _FakeIncident:
    def __init__(self, **kwargs):
        self.id = kwargs.get("id", 1)
        self.workspace_path = kwargs.get("workspace_path", "/repo")
        self.release_id = kwargs.get("release_id", 1)
        self.deployment_strategy = kwargs.get("deployment_strategy", "Rolling")
        self.action = kwargs.get("action", "rolled_back")
        self.root_cause = kwargs.get("root_cause", "The batched writer held a table lock too long.")
        self.created_at = kwargs.get("created_at", datetime(2026, 8, 21, tzinfo=timezone.utc))


# ---------- ingest_project_docs ----------


@pytest.mark.asyncio
async def test_ingest_project_docs_embeds_and_indexes_chunks(tmp_path, monkeypatch):
    import app.rag.ingest as ingest_module

    doc = tmp_path / "design.md"
    doc.write_text("## Section A\nsome content here\n\n## Section B\nmore content\n")
    monkeypatch.setattr(ingest_module, "DOC_FILES", [doc])
    monkeypatch.setattr(ingest_module, "DOCS_HASH_PATH", tmp_path / "docs_hash.txt")

    store = VectorStore(persist_dir=tmp_path / "chroma")
    indexed = await ingest_project_docs(_FakeLLM(), store)

    assert indexed > 0
    assert store.count() == indexed


@pytest.mark.asyncio
async def test_ingest_project_docs_is_idempotent_on_unchanged_content(tmp_path, monkeypatch):
    import app.rag.ingest as ingest_module

    doc = tmp_path / "design.md"
    doc.write_text("## Same heading\nunchanged body\n")
    monkeypatch.setattr(ingest_module, "DOC_FILES", [doc])
    monkeypatch.setattr(ingest_module, "DOCS_HASH_PATH", tmp_path / "docs_hash.txt")

    store = VectorStore(persist_dir=tmp_path / "chroma_idempotent")
    first = await ingest_project_docs(_FakeLLM(), store)
    second = await ingest_project_docs(_FakeLLM(), store)

    assert first > 0
    assert second == 0  # unchanged content -> zero new embedding calls, not a re-embed


@pytest.mark.asyncio
async def test_ingest_project_docs_reembeds_after_content_changes(tmp_path, monkeypatch):
    import app.rag.ingest as ingest_module

    doc = tmp_path / "design.md"
    doc.write_text("## Heading\noriginal body\n")
    monkeypatch.setattr(ingest_module, "DOC_FILES", [doc])
    monkeypatch.setattr(ingest_module, "DOCS_HASH_PATH", tmp_path / "docs_hash.txt")

    store = VectorStore(persist_dir=tmp_path / "chroma_changed")
    await ingest_project_docs(_FakeLLM(), store)

    doc.write_text("## Heading\na genuinely different body this time\n")
    second = await ingest_project_docs(_FakeLLM(), store)

    assert second > 0  # the content hash changed, so this must not be treated as unchanged


# ---------- ingest_release_record / ingest_incident_record ----------


@pytest.mark.asyncio
async def test_ingest_release_record_embeds_and_links_graph_node(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    graph = GraphStore(db_path=tmp_path / "graph.db")
    record = _FakeRelease(id=1, target_file="checkout.py")

    await ingest_release_record(_FakeLLM(), store, graph, record)

    assert store.count() == 1
    facts = graph.recent_relationships("/repo")
    assert any(f["relation"] == "HAS_RELEASE" for f in facts)


@pytest.mark.asyncio
async def test_ingest_incident_record_embeds_and_links_to_its_release(tmp_path):
    """Encodes the exact lesson this project's own live testing surfaced: a CAUSED edge
    only shows up once the release it points from has actually been ingested first —
    so this test ingests the release before the incident, matching production's real
    order in main.py's _maybe_record_release()."""
    store = VectorStore(persist_dir=tmp_path / "chroma")
    graph = GraphStore(db_path=tmp_path / "graph.db")
    release = _FakeRelease(id=5, target_file="checkout.py")
    await ingest_release_record(_FakeLLM(), store, graph, release)

    incident = _FakeIncident(id=1, release_id=5)
    await ingest_incident_record(_FakeLLM(), store, graph, incident)

    assert store.count() == 2  # release chunk + incident chunk
    facts = graph.recent_relationships("/repo")
    assert any(f["relation"] == "CAUSED" for f in facts)
