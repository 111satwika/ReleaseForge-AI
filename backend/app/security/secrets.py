"""Blocks this tool from ever reading a file that's conventionally where real secrets
live — the design doc's explicit Section 14 callout: "Never expose secrets such as
.env, API keys, cloud credentials, tokens, or private keys unnecessarily."

The real risk this closes isn't "an attacker reads your .env" (anyone with filesystem
access already can) — it's that `read_file`'s content gets embedded straight into an
LLM prompt (see graph.py's `code()` node, which sends "current file contents" to the
model) or returned in an API response. A repo task that innocently asks to "update the
config" should never accidentally cause a real API key to be echoed into a chat
transcript or sent to an LLM provider. Filenames are not blocked from `list_files` —
knowing a `.env` file exists isn't a leak, only its contents are.
"""

import fnmatch

# Conventional secret-shaped filenames — deliberately pattern-based (not a fixed list),
# so a project's own `.env.production` or `id_rsa.backup` still matches. Not everything
# with "key" or "secret" in the name is included (e.g. `keymap.json`, a keyboard
# shortcuts file, would false-positive on a bare "*key*") — patterns are anchored to
# real, specific secret-file conventions instead.
SENSITIVE_PATH_PATTERNS = [
    ".env",
    ".env.*",
    # The dotfile-prefix form above (".env", ".env.local") misses the equally common
    # suffix convention — "production.env", "local.env" — no leading dot at all. Found
    # live: a file named "_fake_secret_for_test.env" sailed straight through the guard
    # during Phase 9's live verification because only the prefix form was checked.
    "*.env",
    "*.pem",
    "*.key",
    "*.pfx",
    "*.p12",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "id_ecdsa",
    "id_ecdsa.*",
    "*credentials*.json",
    "*serviceaccount*.json",
    "secrets.yaml",
    "secrets.yml",
    ".npmrc",
    ".pypirc",
    ".netrc",
]

# Checked before the patterns above, not folded into them: `.env.example` matches the
# broad ".env.*" pattern too, but these suffixes are a real, widely-followed convention
# for a *template* committed to the repo on purpose — placeholder keys, not live ones.
# Blocking them would refuse an entirely ordinary task ("update the example env file")
# for no actual safety benefit.
SAFE_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist")


def is_sensitive_path(relative_path: str) -> bool:
    """Matches on the filename component only — `config/.env` and `.env` are both
    caught, but a path merely *containing* a directory named e.g. `secrets/` isn't
    blocked wholesale (a repo can legitimately have a `secrets/README.md` explaining
    its secrets-management approach, which is exactly the kind of file this tool should
    still be able to read and edit)."""
    filename = relative_path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if filename.startswith(".env.") and filename.endswith(SAFE_ENV_TEMPLATE_SUFFIXES):
        return False
    return any(fnmatch.fnmatch(filename, pattern) for pattern in SENSITIVE_PATH_PATTERNS)
