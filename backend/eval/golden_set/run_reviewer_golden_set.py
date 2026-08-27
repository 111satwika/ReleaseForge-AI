"""Runs the Review Agent golden set (dataset.py, 42 defect + 16 clean fixtures) against
the REAL review() node in app/agents/graph.py — the exact production code path, calling
whatever LLMProvider get_llm_provider() returns (Ollama by default), not a mock.

Usage (from backend/):
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --update-champion
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --check-champion-freshness

--update-champion: after a live run, refuses to overwrite champion.json if the new
critical-defect catch rate is more than 5 points below the currently recorded champion
(the same regression gate the CI job enforces for a prompt-file change) — pass
--force-update to overwrite anyway (e.g. accepting a known, deliberate tradeoff).

--check-champion-freshness: no LLM calls, no network — just verifies champion.json's
recorded prompt_hash matches app/prompts/review_agent_v1.txt's current hash. This is the
fast, free check CI actually runs; a live re-score can't run on a hosted GitHub Actions
runner in reasonable time (no Ollama there, and this set alone takes tens of minutes on a
CPU-only local model — see the CI workflow's own comment on why).
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agents.graph import review  # noqa: E402
from app.llm.factory import get_llm_provider  # noqa: E402
from app.prompts import prompt_hash  # noqa: E402
from eval.golden_set.dataset import ReviewFixture, load_golden_set  # noqa: E402

CHAMPION_PATH = Path(__file__).parent / "champion.json"
RESULTS_PATH = Path(__file__).parent / "results" / "latest.json"
PROMPT_NAME = "review_agent_v1"
REGRESSION_TOLERANCE_POINTS = 5.0
CONCURRENCY = 3


async def _run_one(fixture: ReviewFixture, semaphore: asyncio.Semaphore) -> dict:
    state = {
        "change_description": fixture.change_description,
        "target_file": fixture.target_file,
        "original_content": fixture.original_content,
        "new_content": fixture.new_content,
        "test_passed": fixture.test_passed,
        "review_round": 0,
    }
    async with semaphore:
        result = await review(state)

    combined_text = (result.get("review_feedback", "") or "").lower()
    status = result.get("review_status", "APPROVED")

    caught = False
    if fixture.defect_type:
        caught = status == "CHANGES_REQUESTED" and any(kw in combined_text for kw in fixture.catch_keywords)

    false_positive = fixture.defect_type is None and status == "CHANGES_REQUESTED"

    return {
        "id": fixture.id,
        "defect_type": fixture.defect_type,
        "severity": fixture.severity,
        "line_count": fixture.new_content.count("\n") + 1,
        "review_status": status,
        "review_feedback": result.get("review_feedback", ""),
        "caught": caught,
        "false_positive": false_positive,
    }


def _compute_metrics(results: list[dict]) -> dict:
    defect_results = [r for r in results if r["defect_type"]]
    clean_results = [r for r in results if not r["defect_type"]]

    def catch_rate(severity: str | None) -> float | None:
        pool = defect_results if severity is None else [r for r in defect_results if r["severity"] == severity]
        if not pool:
            return None
        return round(100.0 * sum(1 for r in pool if r["caught"]) / len(pool), 1)

    total_positives = len(defect_results)
    true_positives = sum(1 for r in defect_results if r["caught"])
    false_negatives = total_positives - true_positives
    false_positives = sum(1 for r in clean_results if r["false_positive"])
    clean_total_lines = sum(r["line_count"] for r in clean_results)

    precision = round(100.0 * true_positives / (true_positives + false_positives), 1) if (true_positives + false_positives) else None
    recall = round(100.0 * true_positives / (true_positives + false_negatives), 1) if (true_positives + false_negatives) else None
    fp_rate_per_100_lines = round(100.0 * false_positives / clean_total_lines, 2) if clean_total_lines else None

    return {
        "critical_catch_rate": catch_rate("critical"),
        "major_catch_rate": catch_rate("major"),
        "minor_catch_rate": catch_rate("minor"),
        "overall_catch_rate": catch_rate(None),
        "false_positive_rate_per_100_lines": fp_rate_per_100_lines,
        "precision": precision,
        "recall": recall,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "true_negatives": len(clean_results) - false_positives,
        "defect_count": total_positives,
        "clean_count": len(clean_results),
    }


def _print_report(metrics: dict, results: list[dict], model: str) -> None:
    print(f"\n=== Review Agent golden set — {len(results)} fixtures, model={model} ===\n")
    print(f"Critical catch rate: {metrics['critical_catch_rate']}%")
    print(f"Major catch rate:    {metrics['major_catch_rate']}%")
    print(f"Minor catch rate:    {metrics['minor_catch_rate']}%")
    print(f"Overall catch rate:  {metrics['overall_catch_rate']}%")
    print(f"False positive rate: {metrics['false_positive_rate_per_100_lines']} per 100 lines")
    print(f"Precision:           {metrics['precision']}%")
    print(f"Recall:              {metrics['recall']}%")
    print(
        f"TP={metrics['true_positives']} FN={metrics['false_negatives']} "
        f"FP={metrics['false_positives']} TN={metrics['true_negatives']}\n"
    )
    missed = [r for r in results if r["defect_type"] and not r["caught"]]
    if missed:
        print(f"Missed defects ({len(missed)}):")
        for r in missed:
            print(f"  - {r['id']} ({r['severity']}, {r['defect_type']}) -> status={r['review_status']}")
    fps = [r for r in results if r["false_positive"]]
    if fps:
        print(f"\nFalse positives ({len(fps)}):")
        for r in fps:
            print(f"  - {r['id']}")


async def _run_all() -> tuple[dict, list[dict], str]:
    fixtures = load_golden_set()
    llm = get_llm_provider()
    model = getattr(llm, "model", "unknown")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(_run_one(f, semaphore) for f in fixtures))
    metrics = _compute_metrics(results)
    return metrics, results, model


def _check_champion_freshness() -> int:
    if not CHAMPION_PATH.exists():
        print("No champion.json found — nothing to check.")
        return 0
    champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8"))
    if champion.get("prompt_hash") is None:
        print("champion.json has no recorded score yet (prompt_hash is null) — nothing to compare, passing.")
        return 0
    current_hash = prompt_hash(PROMPT_NAME)
    if champion["prompt_hash"] != current_hash:
        print(
            f"FAIL: {PROMPT_NAME} changed (hash {champion['prompt_hash']} -> {current_hash}) but "
            "champion.json wasn't refreshed. Run "
            "`.venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --update-champion` "
            "locally against a real model and commit the result before merging this prompt change."
        )
        return 1
    print("champion.json is fresh — prompt hash matches the recorded baseline.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-champion", action="store_true")
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument("--check-champion-freshness", action="store_true")
    args = parser.parse_args()

    if args.check_champion_freshness:
        return _check_champion_freshness()

    metrics, results, model = asyncio.run(_run_all())
    _print_report(metrics, results, model)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"model": model, "metrics": metrics, "results": results}, indent=2), encoding="utf-8")
    print(f"\nFull results written to {RESULTS_PATH}")

    if args.update_champion:
        champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8")) if CHAMPION_PATH.exists() else {}
        previous_critical = champion.get("critical_catch_rate")
        new_critical = metrics["critical_catch_rate"] or 0.0

        if previous_critical is not None and not args.force_update:
            if new_critical < previous_critical - REGRESSION_TOLERANCE_POINTS:
                print(
                    f"\nREFUSING to update champion.json: critical catch rate dropped from "
                    f"{previous_critical}% to {new_critical}% (more than {REGRESSION_TOLERANCE_POINTS} points). "
                    "Pass --force-update to override if this is a deliberate, accepted tradeoff."
                )
                return 1

        champion_out = {
            "prompt_name": PROMPT_NAME,
            "prompt_hash": prompt_hash(PROMPT_NAME),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            **{k: metrics[k] for k in (
                "critical_catch_rate", "major_catch_rate", "minor_catch_rate", "overall_catch_rate",
                "false_positive_rate_per_100_lines", "precision", "recall",
            )},
        }
        CHAMPION_PATH.write_text(json.dumps(champion_out, indent=2), encoding="utf-8")
        print(f"champion.json updated at {CHAMPION_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
