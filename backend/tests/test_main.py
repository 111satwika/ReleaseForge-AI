"""First endpoint-level tests this project has ever had for any FastAPI route — every
one of these was previously only verified by hand during a build phase (see
RELEASEFORGE_BUILD_PLAN.md). conftest.py points this whole session at an isolated temp
database, so seeding data here never touches this machine's real backend/data/memory.db.

/chat itself still isn't covered: it needs the real LangGraph checkpointer + the MCP
client's three real subprocess servers, both only wired up in main.py's lifespan, and a
real or faithfully-mocked LLM behind every graph node (classify_intent, plan, review,
...) — a materially bigger integration-test investment than the read-only endpoints
below. Left as a named, open gap rather than attempted here just to claim coverage."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import app.main as main_module
from app.memory import repository as memory
from app.memory.db import init_db
from app.rag.chunking import Chunk
from app.rag.vector_store import VectorStore
from app.security import oauth
from app.security.audit import record_audit_event

AUTH = {"Authorization": "Bearer test-token"}


@pytest_asyncio.fixture(autouse=True)
async def _ensure_test_db_ready():
    """conftest.py redirects DB_PATH before this module ever imports it; this just makes
    sure the tables actually exist before a test tries to read/write them. Safe to call
    every test — SQLAlchemy's create_all is a no-op for tables that already exist."""
    await init_db()


@pytest.fixture(autouse=True)
def _authenticated():
    """Every real route below except /health and /auth/* now requires a real bearer
    token (see app/security/oauth.py) — set one directly here rather than running an
    actual GitHub device flow in every test. Reset after, so one test's sign-in state
    never leaks into the next."""
    oauth._current_token = "test-token"
    oauth._current_login = "test-user"
    yield
    oauth.sign_out()


client = TestClient(main_module.app)


def test_health_returns_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_protected_routes_reject_a_request_with_no_auth_header():
    resp = client.get("/eval/summary")
    assert resp.status_code == 401


def test_protected_routes_reject_the_wrong_token():
    resp = client.get("/eval/summary", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_eval_summary_computes_rates_from_seeded_records():
    workspace = "/test-workspace/eval-summary"
    await memory.record_release(workspace_path=workspace, outcome="declined_at_plan")
    await memory.record_release(workspace_path=workspace, outcome="deployed", deployment_strategy="Rolling")
    await memory.record_release(workspace_path=workspace, outcome="rolled_back", deployment_strategy="Rolling")

    resp = client.get("/eval/summary", params={"workspace_path": workspace}, headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["records_evaluated"] == 3
    assert body["agent"]["completion_rate"] == round(1 / 3, 4)  # metrics.py rounds to 4dp
    assert body["deployment"]["deployment_success_rate"] == 0.5
    assert body["deployment"]["rollback_rate"] == 0.5


@pytest.mark.asyncio
async def test_audit_log_returns_seeded_events_for_the_workspace():
    workspace = "/test-workspace/audit-log"
    await record_audit_event("coding", "read_file", "ok", "filesystem", workspace)
    await record_audit_event("coding", "delete_deployment", "scope_denied", None, workspace, detail="not permitted")

    resp = client.get("/audit/log", params={"workspace_path": workspace}, headers=AUTH)
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert len(events) == 2
    statuses = {e["status"] for e in events}
    assert statuses == {"ok", "scope_denied"}


@pytest.mark.asyncio
async def test_incidents_returns_seeded_incidents_for_the_workspace():
    workspace = "/test-workspace/incidents"
    release = await memory.record_release(workspace_path=workspace, outcome="rolled_back", deployment_strategy="Canary")
    await memory.record_incident(
        workspace_path=workspace,
        release_id=release.id,
        deployment_strategy="Canary",
        action="rolled_back",
        root_cause="A missing env var crash-looped the canary pod.",
        deploy_log="canary at 1/3 replicas; readiness failed",
    )

    resp = client.get("/incidents", params={"workspace_path": workspace}, headers=AUTH)
    assert resp.status_code == 200
    incidents = resp.json()["incidents"]
    assert len(incidents) == 1
    assert incidents[0]["action"] == "rolled_back"
    assert "env var" in incidents[0]["root_cause"]


def test_incidents_requires_a_workspace_path():
    resp = client.get("/incidents", headers=AUTH)
    assert resp.status_code == 422  # FastAPI's own validation for a missing required query param


# ---------- /eval/faithfulness ----------


class _FakeFaithfulnessLLM:
    """embed(): deterministic length/word-count vector, same trick test_rag.py already
    uses. chat(): one canned judge response, enough to prove the endpoint actually wires
    real retrieval into judge_faithfulness() rather than skipping it."""

    def __init__(self, chat_response: str):
        self._chat_response = chat_response

    async def embed(self, texts):
        return [[float(len(t)), float(t.count(" "))] for t in texts]

    async def chat(self, messages):
        from app.llm.base import LLMResponse

        return LLMResponse(content=self._chat_response, model="fake-judge")


@pytest.mark.asyncio
async def test_eval_faithfulness_retrieves_real_passages_and_returns_the_judges_verdict(tmp_path, monkeypatch):
    fake_llm = _FakeFaithfulnessLLM(
        '{"faithful": false, "unsupported_claims": ["uses Qdrant"], "explanation": "The passage says Chroma."}'
    )
    store = VectorStore(persist_dir=tmp_path / "chroma")
    chunk = Chunk(id="doc:1", text="This project wraps Chroma instead of Qdrant.", source="test")
    embeddings = await fake_llm.embed([chunk.text])
    store.add([chunk], embeddings)

    monkeypatch.setattr(main_module, "get_llm_provider", lambda: fake_llm)
    monkeypatch.setattr(main_module, "get_vector_store", lambda: store)

    resp = client.post(
        "/eval/faithfulness", json={"question": "what vector db is used?", "answer": "It uses Qdrant."}, headers=AUTH
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["faithful"] is False
    assert body["unsupported_claims"] == ["uses Qdrant"]
    assert body["passages_checked"] >= 1


@pytest.mark.asyncio
async def test_eval_faithfulness_handles_a_faithful_answer(tmp_path, monkeypatch):
    fake_llm = _FakeFaithfulnessLLM('{"faithful": true, "unsupported_claims": [], "explanation": "Matches."}')
    store = VectorStore(persist_dir=tmp_path / "chroma_faithful")
    chunk = Chunk(id="doc:1", text="This project wraps Chroma instead of Qdrant.", source="test")
    store.add([chunk], await fake_llm.embed([chunk.text]))

    monkeypatch.setattr(main_module, "get_llm_provider", lambda: fake_llm)
    monkeypatch.setattr(main_module, "get_vector_store", lambda: store)

    resp = client.post(
        "/eval/faithfulness", json={"question": "what vector db is used?", "answer": "It uses Chroma."}, headers=AUTH
    )
    assert resp.status_code == 200
    assert resp.json()["faithful"] is True


# ---------- /auth/* ----------


def test_auth_device_start_returns_the_real_shape_from_oauth(monkeypatch):
    async def fake_start_device_flow():
        return {
            "device_code": "dc123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 5,
        }

    monkeypatch.setattr(oauth, "start_device_flow", fake_start_device_flow)
    resp = client.post("/auth/device/start")  # deliberately no auth header — nothing is authenticated yet
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_code"] == "ABCD-1234"
    assert body["verification_uri"] == "https://github.com/login/device"


def test_auth_device_poll_returns_pending_without_a_token(monkeypatch):
    async def fake_poll_once(device_code):
        return {"status": "pending"}

    monkeypatch.setattr(oauth, "poll_once", fake_poll_once)
    resp = client.post("/auth/device/poll", json={"device_code": "dc123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "pending"
    assert "token" not in body


def test_auth_device_poll_returns_a_token_once_authenticated(monkeypatch):
    async def fake_poll_once(device_code):
        return {"status": "authenticated", "login": "111satwika"}

    monkeypatch.setattr(oauth, "poll_once", fake_poll_once)
    monkeypatch.setattr(oauth, "issued_token", lambda: "test-token")
    resp = client.post("/auth/device/poll", json={"device_code": "dc123"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "authenticated"
    assert body["token"] == "test-token"


def test_auth_sign_out_actually_clears_the_session():
    resp = client.post("/auth/sign_out")
    assert resp.status_code == 200
    assert oauth.is_authenticated() is False
    # The autouse _authenticated fixture re-signs-in after every test on teardown, so
    # nothing further needs restoring here for the next test.
