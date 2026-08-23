"""Points the whole test session at an isolated temp database instead of this
machine's real backend/data/memory.db, so test_main.py's endpoint tests never read or
write real release/incident/audit history. Must run before app.memory.db is ever
imported anywhere — pytest guarantees conftest.py is loaded before any test module in
the same directory is collected, which is what makes this reliable rather than a race
against import order.

Also disables tracing export (this machine doesn't reliably have a Jaeger collector
running, and even when it does, test runs sending real spans there is just noise)."""

import os
import tempfile
from pathlib import Path

_TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="releaseforge_test_"))
os.environ.setdefault("RELEASEFORGE_DB_PATH", str(_TEST_DATA_DIR / "memory.db"))
os.environ.setdefault("RELEASEFORGE_TRACING_ENABLED", "false")
