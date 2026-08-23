from pathlib import Path

import chromadb

from app.rag.chunking import Chunk

CHROMA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma"
CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)


class VectorStore:
    """Chroma in its embedded/persistent mode — a local directory of files, no server
    process and no Docker, the same "runs on this machine, no infra to stand up" bar
    the rest of the memory layer holds itself to (see ProjectMemory/ReleaseRecord on
    SQLite). Swapping to a real Qdrant deployment later means changing this one class,
    not any caller."""

    def __init__(self, persist_dir: Path = CHROMA_DIR, collection_name: str = "releaseforge"):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(collection_name)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, **c.metadata} for c in chunks],
        )

    def query(self, query_embedding: list[float], k: int = 5, where: dict | None = None) -> list[dict]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            where=where,
        )
        hits = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            hits.append({"id": i, "text": doc, "metadata": meta, "distance": dist})
        return hits

    def get_all(self, where: dict | None = None) -> list[dict]:
        """Every stored chunk's id/text/metadata — the corpus the keyword (BM25) side of
        hybrid retrieval scores, since Chroma itself has no sparse/keyword search."""
        if self._collection.count() == 0:
            return []
        result = self._collection.get(where=where)
        docs = []
        for i, doc, meta in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
            docs.append({"id": i, "text": doc, "metadata": meta})
        return docs

    def count(self) -> int:
        return self._collection.count()
