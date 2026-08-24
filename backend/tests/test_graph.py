"""First unit test for anything in app/agents/graph.py directly — every other node in
this file is orchestration (LLM calls, MCP calls, LangGraph state transitions) that this
project has always live-tested against real infrastructure instead of unit-tested (see
RELEASEFORGE_BUILD_PLAN.md for why). _partial_credit_line() and _test_dockerfile_builds()
are different: _partial_credit_line is genuinely pure logic; _test_dockerfile_builds
takes its MCP client as a plain argument rather than reaching for the module-level
singleton, so a fake one can be passed directly without needing to fake the whole
MCPClientManager machinery."""

import pytest

from app.agents.graph import _partial_credit_line, _test_dockerfile_builds


class _FakeDockerMCP:
    """Records every call and returns a scripted result for run_docker_build — enough to
    verify _test_dockerfile_builds()'s actual logic (pass/fail mapping, and that cleanup
    always runs) without a real Docker daemon."""

    def __init__(self, build_ok: bool, build_log: str):
        self._build_ok = build_ok
        self._build_log = build_log
        self.calls: list[tuple[str, str, dict]] = []

    async def call_tool(self, role, tool_name, arguments):
        self.calls.append((role, tool_name, arguments))
        if tool_name == "run_docker_build":
            return {"ok": self._build_ok, "log": self._build_log}
        if tool_name == "remove_docker_image":
            return {"ok": True, "log": "Untagged"}
        raise AssertionError(f"unexpected tool call: {tool_name}")


def test_partial_credit_full_points_when_ran_and_clean():
    points, line = _partial_credit_line("Security scan", ran=True, clean=True, points_if_clean=10, detail="No findings.")
    assert points == 10
    assert "(+10)" in line
    assert "No findings." in line


def test_partial_credit_zero_when_ran_and_not_clean():
    points, line = _partial_credit_line("Dependency check", ran=True, clean=False, points_if_clean=10, detail="2 vulnerabilities.")
    assert points == 0
    assert "(+0)" in line


def test_partial_credit_half_when_it_could_not_run_at_all():
    """A check that couldn't run is neither "verified clean" nor "known bad" — it should
    score strictly between those two outcomes, not collapse to either one."""
    points, line = _partial_credit_line("Infrastructure readiness", ran=False, clean=False, points_if_clean=10, detail="kubectl isn't installed.")
    assert points == 5
    assert 0 < points < 10
    assert "couldn't run" in line
    assert "kubectl isn't installed." in line


# ---------- Dockerfile build verification ----------
# Regression coverage for a real failure this exact check exists to catch: a live task's
# code round added `COPY src/views/nginx-health ...` referencing a source file that
# didn't exist anywhere in the repo. An LLM review caught a different real problem in the
# same file but not that one — only an actual `docker build` attempt catches this class
# of mistake mechanically, the same "real execution, not an opinion" ground truth
# run_tests already gives application code.


@pytest.mark.asyncio
async def test_dockerfile_build_success_marks_tests_passed():
    mcp = _FakeDockerMCP(build_ok=True, build_log="Successfully built abc123")
    result = await _test_dockerfile_builds("/repo", mcp)
    assert result["test_passed"] is True
    assert "builds successfully" in result["test_output"]


@pytest.mark.asyncio
async def test_dockerfile_build_failure_marks_tests_failed_with_the_real_error():
    mcp = _FakeDockerMCP(build_ok=False, build_log='COPY failed: file not found in build context: src/views/nginx-health')
    result = await _test_dockerfile_builds("/repo", mcp)
    assert result["test_passed"] is False
    assert "COPY failed" in result["test_output"]


@pytest.mark.asyncio
async def test_dockerfile_build_always_cleans_up_the_verification_image():
    """Cleanup must run whether the build succeeded or failed — a throwaway verification
    image left behind on every review round would accumulate on disk forever."""
    for build_ok in (True, False):
        mcp = _FakeDockerMCP(build_ok=build_ok, build_log="...")
        await _test_dockerfile_builds("/repo", mcp)
        tool_names = [call[1] for call in mcp.calls]
        assert tool_names == ["run_docker_build", "remove_docker_image"]
        # Every call here uses the "test" role, not "build" — verifying a Dockerfile
        # change is a testing concern, distinct from build_docker_image()'s later,
        # separate real build for actual deployment.
        assert all(call[0] == "test" for call in mcp.calls)
