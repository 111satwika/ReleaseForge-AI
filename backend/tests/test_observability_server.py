"""Real MCP protocol round-trip tests for the new observability server's Resources and
Prompts — the two MCP primitives this project's build plan named as "not yet" built.
Same InMemoryTransport pattern test_mcp_servers.py already uses for the other three
servers' Tools. conftest.py points the whole test session at an isolated temp database,
so seeding data here never touches this machine's real history."""

import json

import pytest
import pytest_asyncio
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport

from app.memory import repository as memory
from app.memory.db import init_db
from app.mcp_servers.observability_server import server as observability_server
from app.security.audit import record_audit_event


@pytest_asyncio.fixture(autouse=True)
async def _ensure_test_db_ready():
    await init_db()


async def _read_resource(uri: str) -> str:
    async with InMemoryTransport(observability_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.read_resource(uri)
            return "\n".join(c.text for c in result.contents if hasattr(c, "text"))


async def _get_prompt(name: str, arguments: dict) -> str:
    async with InMemoryTransport(observability_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.get_prompt(name, arguments)
            return "\n".join(m.content.text for m in result.messages if hasattr(m.content, "text"))


@pytest.mark.asyncio
async def test_observability_server_registers_expected_resources_and_prompts():
    async with InMemoryTransport(observability_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resources = await session.list_resources()
            prompts = await session.list_prompts()

    resource_uris = {str(r.uri) for r in resources.resources}
    assert resource_uris == {"releaseforge://eval-summary", "releaseforge://audit-log", "releaseforge://incidents"}

    prompt_names = {p.name for p in prompts.prompts}
    assert prompt_names == {"classify_intent_prompt", "recovery_analysis_prompt"}


@pytest.mark.asyncio
async def test_eval_summary_resource_reflects_real_seeded_data():
    await memory.record_release(workspace_path="/obs-test", outcome="deployed", deployment_strategy="Rolling")
    text = await _read_resource("releaseforge://eval-summary")
    data = json.loads(text)
    assert data["records_evaluated"] >= 1
    assert "agent" in data and "deployment" in data


@pytest.mark.asyncio
async def test_audit_log_resource_reflects_real_seeded_events():
    await record_audit_event("coding", "read_file", "ok", "filesystem", "/obs-test-audit")
    text = await _read_resource("releaseforge://audit-log")
    events = json.loads(text)
    assert any(e["workspace_path"] == "/obs-test-audit" and e["status"] == "ok" for e in events)


@pytest.mark.asyncio
async def test_incidents_resource_reflects_real_seeded_incidents():
    release = await memory.record_release(workspace_path="/obs-test-incident", outcome="rolled_back")
    await memory.record_incident(
        workspace_path="/obs-test-incident",
        release_id=release.id,
        deployment_strategy="Canary",
        action="rolled_back",
        root_cause="a real root cause for this test",
    )
    text = await _read_resource("releaseforge://incidents")
    incidents = json.loads(text)
    assert any(i["root_cause"] == "a real root cause for this test" for i in incidents)


@pytest.mark.asyncio
async def test_classify_intent_prompt_matches_graph_pys_real_wording():
    text = await _get_prompt("classify_intent_prompt", {"user_message": "add rate limiting to login"})
    assert "Classify the user's message as exactly one word: TASK or CHAT." in text
    assert "add rate limiting to login" in text


@pytest.mark.asyncio
async def test_recovery_analysis_prompt_includes_all_three_evidence_sections():
    text = await _get_prompt(
        "recovery_analysis_prompt",
        {"deploy_log": "apply failed", "pod_logs": "CrashLoopBackOff", "describe_pods": "OOMKilled"},
    )
    assert "apply failed" in text
    assert "CrashLoopBackOff" in text
    assert "OOMKilled" in text
