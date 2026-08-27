"""Loads the two prompts (review_agent, planner) that graph.py's review() and plan()
nodes render at call time, out of plain .txt template files instead of inline f-strings.

This split exists for one reason: the golden-set eval (backend/eval/golden_set/) needs a
"champion" prompt version to score defect-catch-rate regressions against, and a CI gate
that can watch "did the prompt change" via a file path. Neither is meaningful against an
f-string buried inside graph.py — a path-based CI trigger and a content hash both need the
prompt to actually be its own file.
"""

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def prompt_hash(name: str) -> str:
    """Short content hash, used by the golden-set champion file to detect a prompt edit
    that was never re-scored — see eval/golden_set/champion.json's own "prompt_hash" field."""
    return hashlib.sha256(load_prompt(name).encode("utf-8")).hexdigest()[:16]
