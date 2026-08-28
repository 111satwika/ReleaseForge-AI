"""Runs the Review Agent golden set (dataset.py, 42 defect + 16 clean fixtures) against
the REAL review() node in app/agents/graph.py — the exact production code path, calling
whatever LLMProvider get_llm_provider() returns, not a mock. review() resolves its own
model via RELEASEFORGE_REVIEW_MODEL (default: graph.py's DEFAULT_REVIEW_MODEL) —
deliberately separate from the general RELEASEFORGE_LLM_MODEL used by every other node,
since only the Review Agent's behavior has ever been evaluated against a non-default model.

Usage (from backend/):
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --update-champion
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --check-champion-freshness

    # A/B a candidate prompt version through the real review() code path, without
    # touching graph.py's production default:
    .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --prompt-version review_agent_v3 --results-label v3

    # A/B a different model — no code change needed, review() already reads this:
    RELEASEFORGE_REVIEW_MODEL=qwen2.5:7b-instruct .venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --results-label general_model

--update-champion: after a live run, refuses to overwrite champion.json if the new
critical-defect catch rate is more than 5 points below the currently recorded champion
(the same regression gate the CI job enforces for a prompt-file change) — pass
--force-update to overwrite anyway (e.g. accepting a known, deliberate tradeoff). Only
meant to be used for the run that's actually becoming the new production default —
experimental A/B runs should use --results-label instead so they don't clobber
champion.json/results/latest.json with a config that was only being tested.

--results-label NAME: writes to results/latest_NAME.json instead of results/latest.json,
so multiple experimental configs (different prompt version, different model) can be run
and compared side by side without overwriting each other or the real champion baseline.

--check-champion-freshness: no LLM calls, no network — just verifies champion.json's
recorded prompt_hash matches its recorded prompt_name's current file hash. This is the
fast, free check CI actually runs; a live re-score can't run on a hosted GitHub Actions
runner in reasonable time (no Ollama there, and this set alone takes tens of minutes on a
CPU-only local model — see the CI workflow's own comment on why).
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agents.graph import DEFAULT_REVIEW_MODEL, DEFAULT_REVIEW_PROMPT, review  # noqa: E402
from app.prompts import prompt_hash  # noqa: E402
from eval.golden_set.dataset import ReviewFixture, load_golden_set  # noqa: E402

CHAMPION_PATH = Path(__file__).parent / "champion.json"
RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_PROMPT_NAME = DEFAULT_REVIEW_PROMPT
REGRESSION_TOLERANCE_POINTS = 5.0
CONCURRENCY = 3


async def _run_one(fixture: ReviewFixture, semaphore: asyncio.Semaphore, prompt_name: str) -> dict:
    state = {
        "change_description": fixture.change_description,
        "target_file": fixture.target_file,
        "original_content": fixture.original_content,
        "new_content": fixture.new_content,
        "test_passed": fixture.test_passed,
        "review_round": 0,
    }
    async with semaphore:
        result = await review(state, prompt_name=prompt_name)

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


AGGREGATE_KEYS = (
    "critical_catch_rate", "major_catch_rate", "minor_catch_rate", "overall_catch_rate",
    "false_positive_rate_per_100_lines", "precision", "recall",
)


def _aggregate_runs(all_metrics: list[dict]) -> dict:
    """A single 58-fixture run of this exact same config was directly observed to swing
    critical-defect catch rate by 16.7 points (83.3% / 94.4% / 100.0% across 3 identical
    runs) — larger than the regression gate's own 5-point tolerance. A single run is not a
    trustworthy number for a small, non-deterministic local model; this reports mean/min/max
    across N runs instead of pretending one run is ground truth."""
    aggregated = {}
    for key in AGGREGATE_KEYS:
        values = [m[key] for m in all_metrics if m.get(key) is not None]
        if not values:
            aggregated[key] = {"mean": None, "min": None, "max": None, "n": 0}
        else:
            aggregated[key] = {"mean": round(sum(values) / len(values), 1), "min": min(values), "max": max(values), "n": len(values)}
    return aggregated


def _print_aggregate_report(aggregated: dict, num_runs: int, model: str) -> None:
    print(f"\n=== Review Agent golden set — {num_runs} runs, 58 fixtures each, model={model} ===\n")
    for key in AGGREGATE_KEYS:
        a = aggregated[key]
        if a["mean"] is None:
            print(f"{key}: no data")
        else:
            print(f"{key}: mean={a['mean']}  range=[{a['min']}, {a['max']}]  (n={a['n']})")


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


async def _run_all(prompt_name: str) -> tuple[dict, list[dict], str]:
    fixtures = load_golden_set()
    # Mirrors review()'s own resolution exactly, so the report's "model=" label is never
    # wrong — review() picks its model independently inside the call, this is purely for
    # accurate display/recording here.
    model = os.getenv("RELEASEFORGE_REVIEW_MODEL", DEFAULT_REVIEW_MODEL)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(_run_one(f, semaphore, prompt_name) for f in fixtures))
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
    champion_prompt_name = champion.get("prompt_name", DEFAULT_PROMPT_NAME)
    current_hash = prompt_hash(champion_prompt_name)
    if champion["prompt_hash"] != current_hash:
        print(
            f"FAIL: {champion_prompt_name} changed (hash {champion['prompt_hash']} -> {current_hash}) but "
            "champion.json wasn't refreshed. Run "
            "`.venv/Scripts/python -m eval.golden_set.run_reviewer_golden_set --update-champion` "
            "locally against a real model and commit the result before merging this prompt change."
        )
        return 1
    print("champion.json is fresh — prompt hash matches the recorded baseline.")
    return 0


MIN_RUNS_FOR_CHAMPION = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-champion", action="store_true")
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument("--check-champion-freshness", action="store_true")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_NAME, help="Which app/prompts/<name>.txt to test through the real review() node.")
    parser.add_argument("--results-label", default=None, help="Write to results/latest_<label>.json instead of results/latest.json, for side-by-side A/B comparisons.")
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help=(
            "Run the full 58-fixture set this many times and report mean/min/max instead of a single "
            "number. A single run of this model was directly observed to swing critical catch rate by "
            "16.7 points run-to-run — required to be >= 3 when using --update-champion."
        ),
    )
    args = parser.parse_args()

    if args.check_champion_freshness:
        return _check_champion_freshness()

    if args.update_champion and args.num_runs < MIN_RUNS_FOR_CHAMPION:
        print(
            f"REFUSING: --update-champion requires --num-runs >= {MIN_RUNS_FOR_CHAMPION}. A single run's "
            "numbers for this model are not reliable enough to become the recorded baseline — see "
            "RELEASEFORGE_BUILD_PLAN.md's Round 7 for the measured evidence (83.3% / 94.4% / 100.0% "
            "critical catch rate across 3 identical runs of the same config)."
        )
        return 1

    all_metrics: list[dict] = []
    model = ""
    for run_num in range(1, args.num_runs + 1):
        if args.num_runs > 1:
            print(f"\n--- run {run_num}/{args.num_runs} ---")
        metrics, results, model = asyncio.run(_run_all(args.prompt_version))
        _print_report(metrics, results, model)
        all_metrics.append(metrics)

        results_path = RESULTS_DIR / (
            f"latest_{args.results_label}_run{run_num}.json" if args.results_label and args.num_runs > 1
            else f"latest_{args.results_label}.json" if args.results_label
            else "latest.json" if args.num_runs == 1
            else f"latest_run{run_num}.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps({"model": model, "prompt_name": args.prompt_version, "metrics": metrics, "results": results}, indent=2),
            encoding="utf-8",
        )
        print(f"Full results written to {results_path}")

    aggregated = _aggregate_runs(all_metrics)
    if args.num_runs > 1:
        _print_aggregate_report(aggregated, args.num_runs, model)

    if args.update_champion:
        champion = json.loads(CHAMPION_PATH.read_text(encoding="utf-8")) if CHAMPION_PATH.exists() else {}
        previous_critical = champion.get("critical_catch_rate", {}).get("mean") if isinstance(champion.get("critical_catch_rate"), dict) else champion.get("critical_catch_rate")
        new_critical_mean = aggregated["critical_catch_rate"]["mean"] or 0.0

        if previous_critical is not None and not args.force_update:
            if new_critical_mean < previous_critical - REGRESSION_TOLERANCE_POINTS:
                print(
                    f"\nREFUSING to update champion.json: mean critical catch rate dropped from "
                    f"{previous_critical}% to {new_critical_mean}% (more than {REGRESSION_TOLERANCE_POINTS} points). "
                    "Pass --force-update to override if this is a deliberate, accepted tradeoff."
                )
                return 1

        champion_out = {
            "prompt_name": args.prompt_version,
            "prompt_hash": prompt_hash(args.prompt_version),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "num_runs": args.num_runs,
            **aggregated,
        }
        CHAMPION_PATH.write_text(json.dumps(champion_out, indent=2), encoding="utf-8")
        print(f"\nchampion.json updated at {CHAMPION_PATH} (mean of {args.num_runs} runs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
