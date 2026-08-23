"""Unit tests for app/security/secrets.py — the sensitive-filename guard."""

import pytest

from app.security.secrets import is_sensitive_path


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        "config/.env",
        # Suffix convention (no leading dot) — a real bug caught by Phase 9's live
        # verification: a file literally named "_fake_secret_for_test.env" sailed
        # straight through the guard before this pattern was added.
        "production.env",
        "local.env",
        "_fake_secret_for_test.env",
        "id_rsa",
        "id_rsa.pub",
        ".ssh/id_ed25519",
        "certs/server.pem",
        "keys/private.key",
        "aws-credentials.json",
        "gcp-serviceaccount.json",
        "secrets.yaml",
        ".npmrc",
        ".netrc",
    ],
)
def test_flags_conventionally_secret_shaped_paths(path):
    assert is_sensitive_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "app.py",
        "src/keymap.json",  # contains "key" but isn't a credential file
        "docs/secrets/README.md",  # a directory named "secrets", not a secret file itself
        ".env.example",  # a template with placeholder values, not real secrets — see note below
        "package.json",
    ],
)
def test_does_not_flag_ordinary_or_template_files(path):
    # .env.example is arguably a judgment call — it's flagged as NOT sensitive here since
    # it conventionally holds placeholder keys, not real ones, and blocking it would stop
    # a completely ordinary task ("update the example env template"). Real .env* files
    # with actual values are still caught by the ".env.*" pattern for anything else.
    assert is_sensitive_path(path) is False


def test_matches_on_filename_only_case_insensitively():
    assert is_sensitive_path("SOME/DEEP/PATH/ID_RSA") is True
    assert is_sensitive_path("a/b/c/.ENV") is True
