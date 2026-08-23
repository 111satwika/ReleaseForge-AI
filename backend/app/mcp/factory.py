from app.mcp.client import MCPClientManager

# Unlike get_llm_provider()/get_vector_store(), this can't be a lazy-on-first-use
# singleton: starting it spawns three subprocesses and does an async MCP handshake with
# each, which has to happen once, deliberately, during FastAPI's startup — not
# accidentally triggered mid-request by whichever node happens to need a tool first.
_client: MCPClientManager | None = None


async def start_mcp_client() -> None:
    global _client
    if _client is None:
        _client = MCPClientManager()
        await _client.start()


async def stop_mcp_client() -> None:
    global _client
    if _client is not None:
        await _client.stop()
        _client = None


def get_mcp_client() -> MCPClientManager:
    if _client is None:
        raise RuntimeError("MCP client not started — start_mcp_client() must run during app startup first.")
    return _client
