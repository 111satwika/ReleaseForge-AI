"""Tests that MCPClientManager.call_tool actually enforces least-privilege scoping —
built with a fake session (no real subprocess needed) so this runs as a fast unit test,
proving the *enforcement mechanism* itself: a denied call must never even reach the
underlying MCP session, not just that it eventually raises."""

from unittest.mock import AsyncMock

import pytest

from app.mcp.client import MCPClientManager
from app.security.scopes import ScopeDeniedError


@pytest.fixture(autouse=True)
def _no_real_audit_writes(monkeypatch):
    """record_audit_event does a real SQLite write — replaced with a no-op so this test
    file never touches the actual database, same reasoning test_rag.py's fake embedder
    keeps those tests independent of a real Ollama daemon."""
    monkeypatch.setattr("app.mcp.client.record_audit_event", AsyncMock())


def _client_with_fake_session(tool_name: str, server_name: str = "filesystem") -> tuple[MCPClientManager, AsyncMock]:
    client = MCPClientManager()
    fake_session = AsyncMock()
    client._sessions = {server_name: fake_session}
    client._tool_index = {tool_name: server_name}
    return client, fake_session


@pytest.mark.asyncio
async def test_denied_call_never_reaches_the_underlying_session():
    client, fake_session = _client_with_fake_session("delete_deployment")
    with pytest.raises(ScopeDeniedError):
        await client.call_tool("coding", "delete_deployment", {"namespace": "x", "deployment_name": "y"})
    fake_session.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_allowed_call_does_reach_the_underlying_session():
    client, fake_session = _client_with_fake_session("read_file")
    fake_session.call_tool.return_value = type(
        "R", (), {"is_error": False, "structured_content": {"exists": True, "content": "hi"}, "content": []}
    )()
    result = await client.call_tool("coding", "read_file", {"workspace_path": "/x", "relative_path": "a.txt"})
    fake_session.call_tool.assert_called_once()
    assert result == {"exists": True, "content": "hi"}


@pytest.mark.asyncio
async def test_scope_denial_is_audited():
    client, _ = _client_with_fake_session("delete_deployment")
    audit_spy = AsyncMock()
    import app.mcp.client as client_module

    client_module.record_audit_event = audit_spy
    with pytest.raises(ScopeDeniedError):
        await client.call_tool("coding", "delete_deployment", {"namespace": "x", "deployment_name": "y"})
    audit_spy.assert_awaited_once()
    call_kwargs = audit_spy.await_args
    assert call_kwargs.args[0] == "coding"
    assert call_kwargs.args[1] == "delete_deployment"
    assert call_kwargs.args[2] == "scope_denied"


@pytest.mark.asyncio
async def test_unauthorized_tool_name_gives_a_clear_error_not_a_silent_pass():
    client, fake_session = _client_with_fake_session("read_file")
    with pytest.raises(ScopeDeniedError):
        # "coding" role exists and has some scope, but this tool isn't in it, and it also
        # isn't registered in _tool_index under a role that could ever reach it here —
        # confirms scope is checked before tool-existence, fail-closed either way.
        await client.call_tool("coding", "run_docker_build", {"workspace_path": "/x", "tag": "y"})
    fake_session.call_tool.assert_not_called()
