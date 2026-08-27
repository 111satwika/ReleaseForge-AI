"""Runs the Planner golden set (planner_dataset.py, 18 fixtures) against the REAL plan()
node in app/agents/graph.py, through a real MCPClientManager (3 real subprocess servers)
and a real, isolated scratch SQLite database — the same "real execution, not simulation"
approach run_reviewer_golden_set.py uses for the Review Agent.

Usage (from backend/):
    .venv/Scripts/python -m eval.golden_set.run_planner_golden_set
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

RESULTS_PATH = Path(__file__).parent / "results" / "planner_latest.json"

# Must be set before app.memory.db is ever imported (module-level engine creation reads
# it at import time) — an isolated scratch DB, never the real backend/data/memory.db.
_SCRATCH_DB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_SCRATCH_DB.close()
os.environ["RELEASEFORGE_DB_PATH"] = _SCRATCH_DB.name

from app.agents.graph import plan  # noqa: E402
from app.mcp.factory import start_mcp_client, stop_mcp_client  # noqa: E402
from app.memory.db import engine, init_db  # noqa: E402
from eval.golden_set.planner_dataset import FIXTURES, PlannerFixture  # noqa: E402


def _materialize_workspace(fixture: PlannerFixture) -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory(prefix=f"rf-planner-{fixture.id}-")
    base = Path(tmp.name)
    for rel_path in fixture.workspace_files:
        full = base / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(f"// stub file for {fixture.id}\n", encoding="utf-8")
    return tmp


async def _run_one(fixture: PlannerFixture) -> dict:
    workspace_tmp = _materialize_workspace(fixture)
    try:
        state = {"workspace_path": workspace_tmp.name, "user_message": fixture.task_description}
        result = await plan(state)
    finally:
        workspace_tmp.cleanup()

    target_file = result.get("target_file", "") or ""
    change_description = result.get("change_description", "") or ""
    haystack = f"{target_file} {change_description}".lower()

    failures = []
    if fixture.expected_target_file and target_file != fixture.expected_target_file:
        failures.append(f"expected target_file={fixture.expected_target_file!r}, got {target_file!r}")
    for group in fixture.required_keyword_groups:
        if not any(kw.lower() in haystack for kw in group):
            failures.append(f"missing any of required synonym group {group!r}")
    for kw in fixture.forbidden_keywords:
        if kw.lower() in haystack:
            failures.append(f"contains forbidden keyword {kw!r}")

    return {
        "id": fixture.id,
        "category": fixture.category,
        "target_file": target_file,
        "change_description": change_description,
        "passed": not failures,
        "failures": failures,
    }


def _print_report(results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== Planner golden set — {passed}/{total} passed ({round(100 * passed / total, 1)}%) ===\n")

    categories = sorted({r["category"] for r in results})
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["passed"])
        print(f"{cat}: {cat_passed}/{len(cat_results)}")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  - {r['id']}: target_file={r['target_file']!r}")
            for f in r["failures"]:
                print(f"      {f}")


async def main() -> int:
    await init_db()
    await start_mcp_client()
    try:
        results = [await _run_one(f) for f in FIXTURES]
    finally:
        await stop_mcp_client()
        # Windows keeps the sqlite file handle open until the engine's connection pool is
        # explicitly disposed — without this, unlinking the scratch DB below raises
        # PermissionError. That failure previously happened INSIDE this same finally
        # block, which swallowed the whole run: the report below never printed even
        # though every fixture had already completed. Cleanup failing must never cost the
        # results a real live run just produced, so it's now isolated and non-fatal.
        await engine.dispose()
        try:
            Path(_SCRATCH_DB.name).unlink(missing_ok=True)
        except OSError as exc:
            print(f"(non-fatal: couldn't remove scratch db {_SCRATCH_DB.name}: {exc})")

    _print_report(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r["passed"])
    RESULTS_PATH.write_text(
        json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2), encoding="utf-8"
    )
    print(f"\nFull results written to {RESULTS_PATH}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
