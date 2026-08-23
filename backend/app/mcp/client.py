import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from app.observability.tracing import get_tracer, record_exception
from app.security.audit import record_audit_event
from app.security.scopes import ScopeDeniedError, check_scope

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


def extract_tool_result(result: types.CallToolResult) -> dict | list | str | int | float | bool | None:
    """Turns a raw CallToolResult into the plain value the underlying tools.py/
    k8s_tools.py function actually returned. Pulled out as its own function — not
    inlined in MCPClientManager.call_tool — so tests can exercise this exact unwrapping
    logic directly against a real CallToolResult (from the in-memory transport) without
    needing a full running MCPClientManager.

    Raises RuntimeError if the tool call itself failed."""
    if result.is_error:
        text = "; ".join(c.text for c in result.content if hasattr(c, "text")) or "(no error detail)"
        raise RuntimeError(text)

    if result.structured_content is not None:
        # Tools returning a plain (non-dict/model) value get wrapped as {"result": value}
        # by the SDK's automatic structured-output handling — unwrap that one case so
        # callers get back exactly what the underlying tools.py/k8s_tools.py function
        # itself returned, not an extra layer only MCP's wire format needed.
        if set(result.structured_content.keys()) == {"result"}:
            return result.structured_content["result"]
        return result.structured_content

    # This SDK version doesn't reliably populate structured_content for every plain
    # `dict`-annotated return (observed directly: it did for list/str/bool returns, not
    # for `-> dict`), only its text-content fallback. Every tool here still returns
    # JSON-serializable data, so parse that text as JSON rather than handing callers a
    # pretty-printed string they'd have to parse themselves.
    text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text

# One MCP client orchestrating three MCP servers, each its own subprocess — the design
# doc's Section 5 "multi-server orchestration" concept, made real rather than simulated
# with in-process function calls dressed up as MCP. Each server is launched as a Python
# module (`-m app.mcp_servers.X`) with cwd pinned to backend/, so it resolves the `app`
# package the same way `uvicorn app.main:app` already does.
SERVER_MODULES = {
    "filesystem": "app.mcp_servers.filesystem_server",
    "docker": "app.mcp_servers.docker_server",
    "kubernetes": "app.mcp_servers.kubernetes_server",
}


class MCPClientManager:
    """Started once in FastAPI's lifespan (main.py), stopped on shutdown — the same
    "open expensive resources once, not per request" pattern already used for the
    LangGraph checkpointer. Routes a tool call to whichever server actually registered
    that tool name, discovered from each server's real `list_tools()` response rather
    than a hardcoded mapping — genuine tool discovery, not just a lookup table with
    extra steps."""

    def __init__(self):
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}
        self._tool_index: dict[str, str] = {}  # tool name -> owning server name

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        for server_name, module in SERVER_MODULES.items():
            params = StdioServerParameters(
                command=sys.executable, args=["-m", module], cwd=str(BACKEND_DIR)
            )
            read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._sessions[server_name] = session

            discovered = await session.list_tools()
            for tool in discovered.tools:
                self._tool_index[tool.name] = server_name

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._sessions.clear()
        self._tool_index.clear()

    async def call_tool(
        self, role: str, tool_name: str, arguments: dict
    ) -> dict | list | str | int | float | bool | None:
        """`role` is required, not optional — every call site in graph.py names the
        agent role it's acting as (see app/security/scopes.py), so a new call site can't
        accidentally slip in unscoped just because nobody thought to add the argument."""
        workspace_path = arguments.get("workspace_path")

        try:
            check_scope(role, tool_name)
        except ScopeDeniedError as exc:
            await record_audit_event(role, tool_name, "scope_denied", workspace_path=workspace_path, detail=str(exc))
            raise

        server_name = self._tool_index.get(tool_name)
        if server_name is None:
            await record_audit_event(role, tool_name, "error", workspace_path=workspace_path, detail="unknown tool")
            raise ValueError(
                f"No MCP server exposes a tool named '{tool_name}'. "
                f"Known tools: {sorted(self._tool_index)}"
            )

        tracer = get_tracer()
        with tracer.start_as_current_span(f"mcp.call_tool.{tool_name}") as span:
            span.set_attribute("mcp.tool_name", tool_name)
            span.set_attribute("mcp.server_name", server_name)
            span.set_attribute("mcp.role", role)
            session = self._sessions[server_name]
            try:
                result = await session.call_tool(tool_name, arguments)
                value = extract_tool_result(result)
                await record_audit_event(role, tool_name, "ok", server_name, workspace_path)
                return value
            except RuntimeError as exc:
                wrapped = RuntimeError(f"MCP tool '{tool_name}' on server '{server_name}' failed: {exc}")
                record_exception(span, wrapped)
                await record_audit_event(role, tool_name, "error", server_name, workspace_path, detail=str(exc))
                raise wrapped from exc
            except Exception as exc:
                record_exception(span, exc)
                await record_audit_event(role, tool_name, "error", server_name, workspace_path, detail=str(exc))
                raise

    def available_tools(self) -> dict[str, str]:
        """tool name -> owning server name, for anything that wants to show what's
        discoverable (e.g. a debug endpoint) without hardcoding the server layout."""
        return dict(self._tool_index)
