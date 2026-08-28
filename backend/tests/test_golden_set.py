"""Fast, deterministic tests for the golden-set harness's pure logic — metrics math and
the champion-freshness check — plus dataset structural sanity. The harnesses' own live
paths (calling the real review()/plan() nodes against a real LLM) are deliberately NOT
exercised here, same reasoning as test_tools.py's pip-audit/bandit split: a live run takes
tens of minutes and needs Ollama, neither of which belongs in this suite's normal ~10s
run. Real results are recorded in eval/golden_set/champion.json and
RELEASEFORGE_BUILD_PLAN.md instead."""

import json

import eval.golden_set.run_reviewer_golden_set as reviewer_harness
from eval.golden_set.dataset import load_golden_set
from eval.golden_set.planner_dataset import FIXTURES as PLANNER_FIXTURES


def test_reviewer_dataset_has_expected_shape():
    fixtures = load_golden_set()
    defect = [f for f in fixtures if f.defect_type]
    clean = [f for f in fixtures if not f.defect_type]

    assert len(defect) == 42
    assert len(clean) == 16
    assert len({f.id for f in fixtures}) == len(fixtures), "fixture ids must be unique"

    severities = {f.severity for f in defect}
    assert severities == {"critical", "major", "minor"}
    for f in defect:
        assert f.catch_keywords, f"{f.id} has no catch_keywords to grade against"


def test_planner_dataset_has_expected_shape():
    assert len(PLANNER_FIXTURES) == 18
    assert len({f.id for f in PLANNER_FIXTURES}) == len(PLANNER_FIXTURES)
    for f in PLANNER_FIXTURES:
        assert f.required_keyword_groups or f.forbidden_keywords or f.expected_target_file, (
            f"{f.id} asserts nothing — a fixture with no checks can never fail"
        )


def test_compute_metrics_counts_tp_fn_fp_tn_correctly():
    results = [
        {"defect_type": "sql_injection", "severity": "critical", "caught": True, "false_positive": False, "line_count": 5},
        {"defect_type": "sql_injection", "severity": "critical", "caught": False, "false_positive": False, "line_count": 5},
        {"defect_type": "off_by_one", "severity": "major", "caught": True, "false_positive": False, "line_count": 5},
        {"defect_type": None, "severity": None, "caught": False, "false_positive": True, "line_count": 10},
        {"defect_type": None, "severity": None, "caught": False, "false_positive": False, "line_count": 10},
    ]
    metrics = reviewer_harness._compute_metrics(results)

    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["critical_catch_rate"] == 50.0
    assert metrics["major_catch_rate"] == 100.0
    assert metrics["minor_catch_rate"] is None  # no minor fixtures in this sample — None, not 0
    assert metrics["precision"] == round(100.0 * 2 / 3, 1)
    assert metrics["recall"] == round(100.0 * 2 / 3, 1)
    assert metrics["false_positive_rate_per_100_lines"] == 5.0  # 1 FP / 20 clean lines * 100


def test_compute_metrics_zero_false_positives_gives_zero_not_none():
    results = [
        {"defect_type": "sql_injection", "severity": "critical", "caught": True, "false_positive": False, "line_count": 5},
        {"defect_type": None, "severity": None, "caught": False, "false_positive": False, "line_count": 10},
    ]
    metrics = reviewer_harness._compute_metrics(results)
    assert metrics["false_positive_rate_per_100_lines"] == 0.0
    assert metrics["precision"] == 100.0


def test_check_champion_freshness_passes_when_hash_matches(monkeypatch, tmp_path):
    champion_path = tmp_path / "champion.json"
    real_hash = reviewer_harness.prompt_hash(reviewer_harness.DEFAULT_PROMPT_NAME)
    champion_path.write_text(json.dumps({"prompt_hash": real_hash}), encoding="utf-8")
    monkeypatch.setattr(reviewer_harness, "CHAMPION_PATH", champion_path)

    assert reviewer_harness._check_champion_freshness() == 0


def test_check_champion_freshness_fails_when_prompt_changed_but_champion_stale(monkeypatch, tmp_path):
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps({"prompt_hash": "stale-hash-that-cannot-match"}), encoding="utf-8")
    monkeypatch.setattr(reviewer_harness, "CHAMPION_PATH", champion_path)

    assert reviewer_harness._check_champion_freshness() == 1


def test_check_champion_freshness_passes_when_no_champion_recorded_yet(monkeypatch, tmp_path):
    champion_path = tmp_path / "champion.json"
    champion_path.write_text(json.dumps({"prompt_hash": None}), encoding="utf-8")
    monkeypatch.setattr(reviewer_harness, "CHAMPION_PATH", champion_path)

    assert reviewer_harness._check_champion_freshness() == 0


def test_aggregate_runs_computes_mean_min_max():
    # The exact real observation that motivated this function: 3 identical-config runs of
    # this model swung critical catch rate from 83.3% to 100.0% — a single run isn't a
    # trustworthy number, so the harness reports a range instead of pretending otherwise.
    runs = [
        {"critical_catch_rate": 94.4, "major_catch_rate": 50.0, "minor_catch_rate": 50.0,
         "overall_catch_rate": 69.0, "false_positive_rate_per_100_lines": 0.0, "precision": 100.0, "recall": 69.0},
        {"critical_catch_rate": 83.3, "major_catch_rate": 66.7, "minor_catch_rate": 100.0,
         "overall_catch_rate": 78.6, "false_positive_rate_per_100_lines": 0.96, "precision": 97.1, "recall": 78.6},
        {"critical_catch_rate": 100.0, "major_catch_rate": 66.7, "minor_catch_rate": 0.0,
         "overall_catch_rate": 71.4, "false_positive_rate_per_100_lines": 1.92, "precision": 93.8, "recall": 71.4},
    ]
    aggregated = reviewer_harness._aggregate_runs(runs)

    assert aggregated["critical_catch_rate"]["min"] == 83.3
    assert aggregated["critical_catch_rate"]["max"] == 100.0
    assert aggregated["critical_catch_rate"]["mean"] == round((94.4 + 83.3 + 100.0) / 3, 1)
    assert aggregated["critical_catch_rate"]["n"] == 3


def test_aggregate_runs_skips_none_values_instead_of_treating_as_zero():
    runs = [
        {"minor_catch_rate": None, "critical_catch_rate": 100.0, "major_catch_rate": 50.0,
         "overall_catch_rate": 80.0, "false_positive_rate_per_100_lines": 0.0, "precision": 100.0, "recall": 80.0},
        {"minor_catch_rate": None, "critical_catch_rate": 100.0, "major_catch_rate": 50.0,
         "overall_catch_rate": 80.0, "false_positive_rate_per_100_lines": 0.0, "precision": 100.0, "recall": 80.0},
    ]
    aggregated = reviewer_harness._aggregate_runs(runs)

    assert aggregated["minor_catch_rate"] == {"mean": None, "min": None, "max": None, "n": 0}
    assert aggregated["critical_catch_rate"]["n"] == 2


def test_main_refuses_update_champion_with_too_few_runs(monkeypatch, capsys):
    monkeypatch.setattr(
        reviewer_harness.sys, "argv",
        ["run_reviewer_golden_set.py", "--update-champion", "--num-runs", "1"],
    )
    exit_code = reviewer_harness.main()
    assert exit_code == 1
    assert "REFUSING" in capsys.readouterr().out
