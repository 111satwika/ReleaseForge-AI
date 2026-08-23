import sqlite3
from pathlib import Path

GRAPH_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "graph.db"
GRAPH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src TEXT NOT NULL,
    relation TEXT NOT NULL,
    dst TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    release_id INTEGER,
    UNIQUE(src, relation, dst, release_id)
);
CREATE INDEX IF NOT EXISTS idx_edges_workspace ON edges(workspace_path);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


class GraphStore:
    """A relationship graph over release history, on SQLite — the design doc calls for
    Neo4j/Cypher here, but that needs a server and there's still no Docker on this
    machine to run one (same blocker as Phase 4). Same principle as ProjectMemory
    standing in for a not-yet-available Postgres: a plain nodes/edges schema captures
    the real relationships (Application -> File, Application -> Release, Release ->
    File) now; swapping to real Neo4j later means changing this module's internals,
    not the shape of what callers ask for.

    This is synchronous by design, like the git/docker calls in agents/tools.py — call
    sites in async code must wrap it in asyncio.to_thread, not call it directly."""

    def __init__(self, db_path: Path = GRAPH_DB_PATH):
        self._db_path = db_path
        conn = sqlite3.connect(self._db_path)
        try:
            conn.executescript(SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _upsert_node(self, conn: sqlite3.Connection, node_id: str, node_type: str, label: str) -> None:
        conn.execute(
            "INSERT INTO nodes (id, type, label) VALUES (?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET label = excluded.label",
            (node_id, node_type, label),
        )

    def _add_edge(
        self, conn: sqlite3.Connection, src: str, relation: str, dst: str, workspace_path: str, release_id: int | None
    ) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO edges (src, relation, dst, workspace_path, release_id) VALUES (?, ?, ?, ?, ?)",
            (src, relation, dst, workspace_path, release_id),
        )

    def upsert_release(self, record) -> None:
        """Called once per completed task run (mirrors record_release in memory/repository.py).
        Builds: Application -[HAS_RELEASE]-> Release -[MODIFIED]-> File, Application -[HAS_FILE]-> File."""
        app_id = f"app:{record.workspace_path}"
        release_id_str = f"release:{record.workspace_path}:{record.id}"

        conn = self._connect()
        try:
            self._upsert_node(conn, app_id, "Application", Path(record.workspace_path).name)
            self._upsert_node(
                conn,
                release_id_str,
                "Release",
                f"{record.outcome} @ {record.created_at.isoformat()}",
            )
            self._add_edge(conn, app_id, "HAS_RELEASE", release_id_str, record.workspace_path, record.id)

            if record.target_file:
                file_id = f"file:{record.workspace_path}:{record.target_file}"
                self._upsert_node(conn, file_id, "File", record.target_file)
                self._add_edge(conn, app_id, "HAS_FILE", file_id, record.workspace_path, None)
                self._add_edge(conn, release_id_str, "MODIFIED", file_id, record.workspace_path, record.id)
            conn.commit()
        finally:
            conn.close()

    def upsert_incident(self, record) -> None:
        """Called once per rollback event (mirrors upsert_release). Builds:
        Release -[CAUSED]-> Incident — the design doc's fuller ... -> Deployment ->
        Incident chain, previously unbuildable simply because no incident data existed
        anywhere to link. release_id is only set when the triggering ReleaseRecord is
        known, so the edge is skipped (not fabricated) if it isn't."""
        incident_id = f"incident:{record.workspace_path}:{record.id}"
        conn = self._connect()
        try:
            self._upsert_node(conn, incident_id, "Incident", f"{record.action} @ {record.created_at.isoformat()}")
            if record.release_id is not None:
                release_id_str = f"release:{record.workspace_path}:{record.release_id}"
                self._add_edge(conn, release_id_str, "CAUSED", incident_id, record.workspace_path, record.release_id)
            conn.commit()
        finally:
            conn.close()

    def releases_touching_file(self, workspace_path: str, target_file: str, limit: int = 10) -> list[dict]:
        """Which past releases touched this exact file — e.g. "who last changed login.py
        and what happened"."""
        file_id = f"file:{workspace_path}:{target_file}"
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT n.label FROM edges e JOIN nodes n ON n.id = e.src "
                "WHERE e.dst = ? AND e.relation = 'MODIFIED' ORDER BY e.release_id DESC LIMIT ?",
                (file_id, limit),
            ).fetchall()
            return [{"release": row[0]} for row in rows]
        finally:
            conn.close()

    def recent_relationships(self, workspace_path: str, limit: int = 8) -> list[dict]:
        """A quick, human-readable slice of this repo's known graph — used to ground chat
        answers without a full Cypher-style traversal query language."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT e.relation, ns.label, nd.label FROM edges e "
                "JOIN nodes ns ON ns.id = e.src JOIN nodes nd ON nd.id = e.dst "
                "WHERE e.workspace_path = ? ORDER BY e.release_id DESC LIMIT ?",
                (workspace_path, limit),
            ).fetchall()
            return [{"relation": r[0], "from": r[1], "to": r[2]} for r in rows]
        finally:
            conn.close()
