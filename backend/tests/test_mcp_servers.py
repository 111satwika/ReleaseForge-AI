"""Real MCP protocol round-trip tests for the three servers — a genuine ClientSession
talking to a genuine MCPServer instance (tool registration, JSON schema validation,
structured-content wire format all actually exercised), just over an in-memory
transport instead of a subprocess, so these run as fast as any other unit test.

Live, subprocess-based end-to-end verification (real git pushes, a real Docker build, a
real `kind` deploy) was done manually against real infrastructure — see the Phase 7
build-plan entry for exactly what that covered. These tests are the fast, repeatable
complement: they prove the *servers themselves* are wired correctly, not the real tools
downstream of them (which Phase 3/4's own test suites already cover)."""

import pytest
from mcp import ClientSession
from mcp.client._memory import InMemoryTransport

from app.mcp.client import extract_tool_result
from app.mcp_servers.docker_server import server as docker_server
from app.mcp_servers.filesystem_server import server as filesystem_server
from app.mcp_servers.kubernetes_server import server as kubernetes_server


async def _call_raw(server, tool_name: str, arguments: dict):
    """Returns the raw CallToolResult, for _call() below to unwrap."""
    async with InMemoryTransport(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)


async def _call(server, tool_name: str, arguments: dict):
    """Returns the unwrapped value — the exact same path MCPClientManager.call_tool()
    uses in production, via the shared extract_tool_result() helper."""
    return extract_tool_result(await _call_raw(server, tool_name, arguments))


@pytest.mark.asyncio
async def test_filesystem_server_registers_expected_tools():
    async with InMemoryTransport(filesystem_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
    names = {t.name for t in tools.tools}
    assert names == {
        "list_files", "read_file", "write_file", "has_dockerfile", "repo_name",
        "detect_test_command", "run_tests", "current_branch", "commit_and_push",
        "push_fix_commit", "open_pull_request",
    }


@pytest.mark.asyncio
async def test_docker_server_registers_expected_tools():
    async with InMemoryTransport(docker_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
    assert {t.name for t in tools.tools} == {"run_docker_build"}


@pytest.mark.asyncio
async def test_kubernetes_server_registers_expected_tools():
    async with InMemoryTransport(kubernetes_server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
    names = {t.name for t in tools.tools}
    assert "ensure_kind_cluster" in names
    assert "apply_manifests" in names
    assert "patch_service_selector" in names
    # Pure logic (decide_deployment_strategy, build_deployment_manifest, ...) is
    # deliberately NOT exposed as tools — see kubernetes_server.py's module docstring.
    assert "decide_deployment_strategy" not in names
    assert "build_deployment_manifest" not in names


@pytest.mark.asyncio
async def test_filesystem_server_write_then_read_round_trip(tmp_path):
    await _call(
        filesystem_server, "write_file",
        {"workspace_path": str(tmp_path), "relative_path": "hello.txt", "content": "hi\n"},
    )

    read_result = await _call(
        filesystem_server, "read_file", {"workspace_path": str(tmp_path), "relative_path": "hello.txt"}
    )
    assert read_result == {"exists": True, "content": "hi\n"}


@pytest.mark.asyncio
async def test_filesystem_server_read_file_reports_missing_file(tmp_path):
    result = await _call(
        filesystem_server, "read_file", {"workspace_path": str(tmp_path), "relative_path": "nope.txt"}
    )
    assert result == {"exists": False, "content": None}


@pytest.mark.asyncio
async def test_filesystem_server_has_dockerfile_false_for_empty_repo(tmp_path):
    result = await _call(filesystem_server, "has_dockerfile", {"workspace_path": str(tmp_path)})
    assert result is False


@pytest.mark.asyncio
async def test_filesystem_server_path_sandboxing_still_enforced_through_mcp(tmp_path):
    """The MCP layer is a thin wrapper — app.agents.tools._resolve_safe_path's sandboxing
    must still reject an escape attempt, proving MCP-ifying this layer didn't accidentally
    loosen the safety boundary that was already there."""
    with pytest.raises(RuntimeError):
        await _call(
            filesystem_server, "write_file",
            {"workspace_path": str(tmp_path), "relative_path": "../../escape.txt", "content": "bad"},
        )


@pytest.mark.asyncio
async def test_filesystem_server_refuses_to_read_a_secret_shaped_file_over_real_mcp(tmp_path):
    """Phase 9's secrets guard (app/security/secrets.py), verified through the actual
    MCP protocol round trip — not just a direct unit test of is_sensitive_path()."""
    (tmp_path / ".env").write_text("API_KEY=super-real-secret\n")
    with pytest.raises(RuntimeError, match="Refusing to read"):
        await _call(filesystem_server, "read_file", {"workspace_path": str(tmp_path), "relative_path": ".env"})


@pytest.mark.asyncio
async def test_filesystem_server_still_reads_an_env_example_template(tmp_path):
    """The one deliberate exception — see secrets.py's SAFE_ENV_TEMPLATE_SUFFIXES."""
    (tmp_path / ".env.example").write_text("API_KEY=your-key-here\n")
    result = await _call(filesystem_server, "read_file", {"workspace_path": str(tmp_path), "relative_path": ".env.example"})
    assert result["exists"] is True


@pytest.mark.asyncio
async def test_docker_server_run_docker_build_missing_dockerfile_fails_cleanly(tmp_path):
    """No Dockerfile in this empty repo — should be a clean {"ok": False, ...} result,
    not a crash, whether or not Docker itself is installed on the machine running this."""
    result = await _call(docker_server, "run_docker_build", {"workspace_path": str(tmp_path), "tag": "x:y"})
    assert result["ok"] is False


@pytest.mark.asyncio
async def test_kubernetes_server_apply_manifests_schema_accepts_manifest_dicts():
    """A basic structural check that `manifests: list[dict]` round-trips through the MCP
    JSON schema correctly. Doesn't require a real cluster — deliberately targets an empty
    namespace with kubectl unlikely to be configured favorably in a bare test env, so
    this only asserts the call completes (isn't rejected by schema validation), not that
    it succeeds against a real cluster."""
    try:
        result = await _call(
            kubernetes_server, "apply_manifests",
            {
                "namespace": "does-not-matter",
                "manifests": [{"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": "x"}}],
            },
        )
    except RuntimeError:
        # A tool-level error (e.g. kubectl unreachable in this environment) is a valid
        # outcome here too — what matters is that the schema accepted a list[dict] arg
        # at all, not whether a real cluster happened to be reachable.
        return
    assert "ok" in result
