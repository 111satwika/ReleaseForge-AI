import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # e.g. "releaseforge-docs", "release-history"
    metadata: dict = field(default_factory=dict)


def _chunk_id(source: str, text: str) -> str:
    # Content-addressed, not sequential — re-ingesting identical text (e.g. an unchanged
    # doc on every backend restart) always produces the same id, so it upserts instead of
    # duplicating, with no separate "have I seen this before" bookkeeping needed.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def chunk_markdown(text: str, source: str, max_chars: int = 1400) -> list[Chunk]:
    """Splits on markdown headings first (keeps a section's ideas together), then
    hard-wraps any section still over max_chars on paragraph boundaries. A README-style
    doc chunks far better on its own structure than on a fixed-size sliding window."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        if re.match(r"^#{1,6}\s+\S", line):
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    chunks: list[Chunk] = []
    for heading, body in sections:
        if not body:
            continue
        pieces = _split_to_budget(body, max_chars) if len(body) > max_chars else [body]
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=_chunk_id(source, piece),
                    text=piece,
                    source=source,
                    metadata={"heading": heading or "(untitled)", "part": i},
                )
            )
    return chunks


def _split_to_budget(body: str, max_chars: int) -> list[str]:
    paragraphs = body.split("\n\n")
    pieces: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_release_record(record) -> Chunk:
    """One release/task run -> one chunk. These are already small (a handful of fields),
    so — unlike a doc — splitting further would only fragment a single fact across
    multiple retrievable pieces for no benefit."""
    lines = [
        f"Release event in {record.workspace_path}",
        f"Outcome: {record.outcome}",
    ]
    if record.target_file:
        lines.append(f"File changed: {record.target_file}")
    if record.change_description:
        lines.append(f"Change: {record.change_description}")
    if record.test_passed is not None:
        lines.append(f"Tests: {'passed' if record.test_passed else 'failed'}")
    if record.review_status:
        lines.append(f"Review: {record.review_status} (risk: {record.review_risk or 'Unknown'})")
    if record.readiness_score is not None:
        lines.append(f"Release readiness: {record.readiness_score}/100 ({record.readiness_recommendation})")
    if record.pr_url:
        lines.append(f"PR: {record.pr_url}")
    lines.append(f"When: {record.created_at.isoformat()}")

    text = "\n".join(lines)
    return Chunk(
        id=f"release-history:{record.id}",
        text=text,
        source="release-history",
        metadata={
            "workspace_path": record.workspace_path,
            "release_id": record.id,
            "outcome": record.outcome,
            "target_file": record.target_file or "",
        },
    )


def chunk_incident_record(record) -> Chunk:
    """One rollback event -> one chunk, same reasoning as chunk_release_record: already
    small, so no further splitting. Makes a past root cause answerable in chat ("why did
    the last deploy roll back") instead of existing only as a one-time reply."""
    lines = [
        f"Incident in {record.workspace_path}",
        f"Action taken: {record.action}",
    ]
    if record.deployment_strategy:
        lines.append(f"Deployment strategy: {record.deployment_strategy}")
    lines.append(f"Root cause: {record.root_cause}")
    lines.append(f"When: {record.created_at.isoformat()}")

    text = "\n".join(lines)
    return Chunk(
        id=f"incident-history:{record.id}",
        text=text,
        source="incident-history",
        metadata={
            "workspace_path": record.workspace_path,
            "incident_id": record.id,
            "release_id": record.release_id or "",
            "action": record.action,
        },
    )
