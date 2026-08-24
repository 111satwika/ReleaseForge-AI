"""MCP server exposing repo-level file and git/GitHub operations — the "Filesystem" and
"Git"/"GitHub" servers from the design doc's Section 5 MCP architecture, combined into
one process since they both operate on the same workspace path and neither is large
enough alone to justify its own subprocess.

This is a thin MCP wrapper: every tool below just calls straight into
`app.agents.tools`, which still owns the actual logic (path sandboxing, subprocess
timeouts, credential-prompt handling). MCP-ifying this layer is about *how the agent
graph reaches these operations* (tool discovery + a protocol boundary instead of a
plain Python import), not a rewrite of what they do.
"""

from mcp.server.mcpserver import MCPServer

from app.agents import tools
from app.security.secrets import is_sensitive_path

server = MCPServer(
    name="releaseforge-filesystem",
    instructions=(
        "Sandboxed file read/write and git/GitHub operations for a single repository "
        "workspace. Every path is validated to stay inside workspace_path."
    ),
)


@server.tool()
def list_files(workspace_path: str, limit: int = 60) -> list[str]:
    """List files in the repo (skips .git/, node_modules/, etc.), for planning context."""
    return tools.list_files(workspace_path, limit=limit)


@server.tool()
def read_file(workspace_path: str, relative_path: str) -> dict:
    """Read one file's contents. Returns {"exists": false} if the file isn't there.

    Refuses conventionally secret-shaped files (.env, *.pem, id_rsa, ...) outright — see
    app/security/secrets.py's module docstring for why: this is the one place a real
    secret's contents could end up inside an LLM prompt or an API response."""
    if is_sensitive_path(relative_path):
        raise ValueError(
            f"Refusing to read '{relative_path}' — it matches a conventionally "
            "secret-shaped filename (.env, key/credential file, ...). If this file "
            "genuinely isn't a secret, rename it away from that convention."
        )
    content = tools.read_file(workspace_path, relative_path)
    if content is None:
        return {"exists": False, "content": None}
    return {"exists": True, "content": content}


@server.tool()
def write_file(workspace_path: str, relative_path: str, content: str) -> dict:
    """Write (or create) a file's full contents."""
    tools.write_file(workspace_path, relative_path, content)
    return {"written": True, "path": relative_path}


@server.tool()
def has_dockerfile(workspace_path: str) -> bool:
    return tools.has_dockerfile(workspace_path)


@server.tool()
def repo_name(workspace_path: str) -> str:
    return tools.repo_name(workspace_path)


@server.tool()
def detect_test_command(workspace_path: str) -> dict:
    """Guesses the repo's test command from recognizable project files (pytest.ini,
    manage.py, package.json's "test" script). Returns {"command": null} if none found."""
    command = tools.detect_test_command(workspace_path)
    return {"command": command}


@server.tool()
def run_tests(workspace_path: str, command: list[str] | None = None) -> dict:
    """Runs the given command (or auto-detects one if omitted). "passed" is null when no
    test command could be found or run at all — distinct from a real test failure."""
    passed, output = tools.run_tests_with_command(workspace_path, command)
    return {"passed": passed, "output": output}


@server.tool()
def check_dependencies(workspace_path: str) -> dict:
    """Real dependency-vulnerability check (pip-audit / npm audit, whichever manifest
    the repo actually has) — closes one of the three gaps release_readiness() used to
    name as "not built yet." "ran": false means the check itself didn't happen (no
    manifest, tool missing, or it errored) — distinct from "ran": true with zero
    vulnerabilities found."""
    return tools.check_dependencies(workspace_path)


@server.tool()
def run_security_scan(workspace_path: str) -> dict:
    """Real static-analysis security scan (bandit, Python-only — see tools.py's
    docstring for why). Same "ran" vs. "found nothing" distinction as check_dependencies."""
    return tools.run_security_scan(workspace_path)


@server.tool()
def current_branch(workspace_path: str) -> str:
    return tools.current_branch(workspace_path)


@server.tool()
def commit_and_push(workspace_path: str, branch_name: str, files: list[str], commit_message: str) -> dict:
    """Creates a new branch, commits the given files, and pushes it to origin."""
    ok, log = tools.commit_and_push(workspace_path, branch_name, files, commit_message)
    return {"ok": ok, "log": log}


@server.tool()
def push_fix_commit(workspace_path: str, files: list[str], commit_message: str) -> dict:
    """Adds a follow-up commit to whatever branch is already checked out, and pushes it."""
    ok, log = tools.push_fix_commit(workspace_path, files, commit_message)
    return {"ok": ok, "log": log}


@server.tool()
def open_pull_request(workspace_path: str, branch_name: str, base_branch: str, title: str, body: str) -> dict:
    """Opens a real PR via `gh`, or falls back to a GitHub compare URL if `gh` isn't installed."""
    ok, result = tools.open_pull_request(workspace_path, branch_name, base_branch, title, body)
    return {"ok": ok, "result": result}


if __name__ == "__main__":
    server.run()
