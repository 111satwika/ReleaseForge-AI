import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class KeywordStore:
    """BM25 over whatever documents are handed in — rebuilt per query rather than kept
    as a standing index. At the corpus sizes this project actually has (docs + one
    project's release history), reindexing on every call is microseconds, and it avoids
    a second store that has to be kept in sync with Chroma's; Chroma's `get_all()` is
    already the single source of truth for chunk text."""

    def __init__(self, documents: list[dict]):
        self._documents = documents
        self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in documents]) if documents else None

    def query(self, text: str, k: int = 5) -> list[dict]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._documents, scores), key=lambda pair: pair[1], reverse=True)
        return [{**doc, "score": score} for doc, score in ranked[:k] if score > 0]
