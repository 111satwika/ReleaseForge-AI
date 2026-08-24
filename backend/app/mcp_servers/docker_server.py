"""MCP server exposing Docker build operations — the "Docker" server from the design
doc's Section 5 MCP architecture. Separate process from the filesystem/git server on
purpose: it demonstrates real multi-server orchestration (the client routes a
`run_docker_build` call to *this* server and a `write_file` call to a different one,
transparently), and matches the natural "give an agent only the tools it needs"
permission boundary from Section 14 — a Coding Agent that only ever needs filesystem+git
tools has no reason to also hold a live handle into the Docker socket.
"""

from mcp.server.mcpserver import MCPServer

from app.agents import tools

server = MCPServer(
    name="releaseforge-docker",
    instructions="Builds a Docker image from a repository's Dockerfile.",
)


@server.tool()
def run_docker_build(workspace_path: str, tag: str) -> dict:
    """Runs `docker build -t <tag> .` in the given workspace. Fails cleanly with a clear
    message (not a crash) if Docker isn't installed."""
    ok, log = tools.run_docker_build(workspace_path, tag)
    return {"ok": ok, "log": log}


@server.tool()
def remove_docker_image(tag: str) -> dict:
    """Removes a tagged image — cleanup for throwaway verification builds, so a Dockerfile
    change's real-build check (see graph.py's test()) doesn't leave images accumulating
    on disk forever."""
    ok, log = tools.remove_docker_image(tag)
    return {"ok": ok, "log": log}


if __name__ == "__main__":
    server.run()
