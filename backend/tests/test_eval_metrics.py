"""Unit tests for the pure metric-computation half of app/eval/metrics.py — no database
needed, just hand-built records with the same shape ReleaseRecord/IncidentRecord rows
have."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.eval.metrics import compute_agent_metrics, compute_coding_metrics, compute_deployment_metrics, compute_mttr_seconds


@dataclass
class FakeRecord:
    outcome: str
    test_passed: bool | None = None
    review_status: str | None = None
    readiness_score: int | None = None
    deployment_strategy: str | None = None


def test_agent_metrics_on_empty_history_is_honestly_none_not_zero():
    metrics = compute_agent_metrics([])
    assert metrics.total_tasks == 0
    assert metrics.completion_rate is None


def test_agent_metrics_buckets_outcomes_correctly():
    records = [
        FakeRecord(outcome="deployed"),
        FakeRecord(outcome="deployed"),
        FakeRecord(outcome="declined_at_plan"),
        FakeRecord(outcome="docker_build_failed"),
        FakeRecord(outcome="rolled_back"),
    ]
    metrics = compute_agent_metrics(records)
    assert metrics.total_tasks == 5
    assert metrics.completed == 2
    assert metrics.declined_by_user == 1
    assert metrics.failed == 1
    assert metrics.rolled_back == 1
    assert metrics.completion_rate == 0.4
    assert metrics.outcome_breakdown == {
        "deployed": 2, "declined_at_plan": 1, "docker_build_failed": 1, "rolled_back": 1,
    }


def test_coding_metrics_ignores_records_with_no_test_or_review_signal():
    records = [
        FakeRecord(outcome="declined_at_plan"),  # never reached test or review
        FakeRecord(outcome="deployed", test_passed=True, review_status="APPROVED", readiness_score=100),
        FakeRecord(outcome="review_unresolved", test_passed=False, review_status="CHANGES_REQUESTED"),
    ]
    metrics = compute_coding_metrics(records)
    assert metrics.tests_run == 2
    assert metrics.tests_passed == 1
    assert metrics.test_pass_rate == 0.5
    assert metrics.reviews_run == 2
    assert metrics.reviews_approved == 1
    assert metrics.review_acceptance_rate == 0.5
    assert metrics.readiness_scores_recorded == 1
    assert metrics.average_readiness_score == 100.0


def test_coding_metrics_all_none_when_nothing_recorded():
    metrics = compute_coding_metrics([FakeRecord(outcome="declined_at_plan")])
    assert metrics.test_pass_rate is None
    assert metrics.review_acceptance_rate is None
    assert metrics.average_readiness_score is None


def test_deployment_metrics_only_counts_records_that_reached_deploy():
    records = [
        FakeRecord(outcome="declined_at_plan"),  # never got near deployment — excluded
        FakeRecord(outcome="deployed", deployment_strategy="Rolling"),
        FakeRecord(outcome="deployed", deployment_strategy="Canary"),
        FakeRecord(outcome="rolled_back", deployment_strategy="Rolling"),
        FakeRecord(outcome="declined_at_rollback", deployment_strategy="Blue-Green"),
    ]
    metrics = compute_deployment_metrics(records)
    assert metrics.deployments_attempted == 4
    assert metrics.deployments_succeeded == 2
    assert metrics.deployment_success_rate == 0.5
    assert metrics.rollbacks == 1
    assert metrics.rollback_rate == 0.25
    assert metrics.strategy_breakdown["Rolling"] == {"attempted": 2, "succeeded": 1, "rolled_back": 1}
    assert metrics.strategy_breakdown["Canary"] == {"attempted": 1, "succeeded": 1, "rolled_back": 0}


def test_deployment_metrics_groups_missing_strategy_as_unknown():
    """A pre-Phase-8 ReleaseRecord row has no deployment_strategy column value at all —
    must not crash, and must still be counted somewhere visible rather than silently
    dropped."""
    records = [FakeRecord(outcome="deployed", deployment_strategy=None)]
    metrics = compute_deployment_metrics(records)
    assert metrics.strategy_breakdown["unknown"]["attempted"] == 1


@dataclass
class FakeIncident:
    created_at: datetime
    deploy_started_at: datetime | None


def test_mttr_is_none_with_no_incidents():
    assert compute_mttr_seconds([]) is None


def test_mttr_averages_real_recovery_durations():
    base = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    incidents = [
        FakeIncident(created_at=base + timedelta(seconds=90), deploy_started_at=base),
        FakeIncident(created_at=base + timedelta(seconds=30), deploy_started_at=base),
    ]
    assert compute_mttr_seconds(incidents) == 60.0  # (90 + 30) / 2


def test_mttr_excludes_incidents_with_no_deploy_started_at_instead_of_treating_as_zero():
    """An incident recorded before deploy_started_at existed has it as None — must be
    excluded outright, not silently counted as an instant 0-second recovery, which would
    quietly drag the average down and misrepresent real recovery speed."""
    base = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    incidents = [
        FakeIncident(created_at=base + timedelta(seconds=100), deploy_started_at=base),
        FakeIncident(created_at=base, deploy_started_at=None),
    ]
    assert compute_mttr_seconds(incidents) == 100.0  # not (100 + 0) / 2 == 50.0


def test_deployment_metrics_includes_mttr_when_incidents_are_passed():
    base = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    records = [FakeRecord(outcome="rolled_back", deployment_strategy="Rolling")]
    incidents = [FakeIncident(created_at=base + timedelta(seconds=45), deploy_started_at=base)]
    metrics = compute_deployment_metrics(records, incidents)
    assert metrics.mean_time_to_recovery_seconds == 45.0
