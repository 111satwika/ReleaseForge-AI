from app.rag.graph_store import GraphStore
from app.rag.vector_store import VectorStore

# Process-wide singletons, same pattern as memory/db.py's module-level `engine` — both
# wrap local, file-backed clients (Chroma's persist dir, a SQLite file) that are safe to
# share across requests, and neither belongs in AgentState: LangGraph's checkpointer
# serializes state to persist paused approvals, and a live Chroma/sqlite3 handle can't
# survive that round-trip.
_vector_store: VectorStore | None = None
_graph_store: GraphStore | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_graph_store() -> GraphStore:
    global _graph_store
    if _graph_store is None:
        _graph_store = GraphStore()
    return _graph_store
