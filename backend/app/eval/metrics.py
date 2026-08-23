"""Platform evaluation metrics — the design doc's Section 16 "Agents"/"Coding"/
"Deployment" categories, computed from real `ReleaseRecord` history rather than
simulated. If the platform hasn't been used yet, every rate here is honestly `None`
(not 0.0, not 100%) — there is a real difference between "0 out of 0 attempts
succeeded" and "10 out of 10 failed," and collapsing that distinction to a fake number
would be exactly the kind of dishonest metric this project has avoided everywhere else.

The pure computation here (`compute_*`) takes a list of already-fetched records and has
no I/O of its own — the same split RAG's chunking/retrieval logic uses — so it's fully
unit-testable without a database.
"""

from dataclasses import dataclass, field

from app.memory import repository as memory

# Every outcome value graph.py's terminal nodes actually set (see ReleaseRecord's
# `outcome` column docstring) — kept in sync there, not re-derived from a DB scan.
SUCCESS_OUTCOMES = {"deployed"}
DECLINED_OUTCOMES = {
    "declined_at_plan",
    "declined_at_code",
    "declined_at_test",
    "declined_at_readiness",
    "declined_at_deploy",
    "declined_at_rollback",
}
FAILED_OUTCOMES = {"publish_failed", "docker_build_failed", "review_unresolved", "planning_failed"}
ROLLED_BACK_OUTCOMES = {"rolled_back"}
DEPLOY_ATTEMPTED_OUTCOMES = {"deployed", "rolled_back", "declined_at_rollback"}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


@dataclass
class AgentMetrics:
    """Design doc Section 16 "Agents": task success rate. Planning accuracy and tool
    selection accuracy aren't included — this project has no ground-truth labels for
    "was that the right file to plan against" or "was that the right tool," so a real
    number for either would have to be fabricated. Not built rather than faked."""

    total_tasks: int
    completed: int
    declined_by_user: int
    failed: int
    rolled_back: int
    completion_rate: float | None
    outcome_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class CodingMetrics:
    """Design doc Section 16 "Coding": tests passed, review acceptance rate. Regression
    rate isn't included — this project has no notion of "this release broke something a
    previous one didn't" to measure against."""

    tests_run: int
    tests_passed: int
    test_pass_rate: float | None
    reviews_run: int
    reviews_approved: int
    review_acceptance_rate: float | None
    readiness_scores_recorded: int
    average_readiness_score: float | None


@dataclass
class DeploymentMetrics:
    """Design doc Section 16 "Deployment": deployment success rate, rollback rate, and —
    since IncidentRecord started stamping deploy_started_at — a real mean time to
    recovery, not inferred by subtracting two timestamps written in the same request
    (which is all that was available before)."""

    deployments_attempted: int
    deployments_succeeded: int
    deployment_success_rate: float | None
    rollbacks: int
    rollback_rate: float | None
    mean_time_to_recovery_seconds: float | None
    strategy_breakdown: dict[str, dict[str, int]] = field(default_factory=dict)


def compute_agent_metrics(records: list) -> AgentMetrics:
    total = len(records)
    breakdown: dict[str, int] = {}
    completed = declined = failed = rolled_back = 0
    for r in records:
        breakdown[r.outcome] = breakdown.get(r.outcome, 0) + 1
        if r.outcome in SUCCESS_OUTCOMES:
            completed += 1
        elif r.outcome in DECLINED_OUTCOMES:
            declined += 1
        elif r.outcome in FAILED_OUTCOMES:
            failed += 1
        elif r.outcome in ROLLED_BACK_OUTCOMES:
            rolled_back += 1
    return AgentMetrics(
        total_tasks=total,
        completed=completed,
        declined_by_user=declined,
        failed=failed,
        rolled_back=rolled_back,
        completion_rate=_rate(completed, total),
        outcome_breakdown=breakdown,
    )


def compute_coding_metrics(records: list) -> CodingMetrics:
    tests_run = tests_passed = 0
    reviews_run = reviews_approved = 0
    readiness_scores: list[int] = []
    for r in records:
        if r.test_passed is not None:
            tests_run += 1
            if r.test_passed:
                tests_passed += 1
        if r.review_status:
            reviews_run += 1
            if r.review_status == "APPROVED":
                reviews_approved += 1
        if r.readiness_score is not None:
            readiness_scores.append(r.readiness_score)
    return CodingMetrics(
        tests_run=tests_run,
        tests_passed=tests_passed,
        test_pass_rate=_rate(tests_passed, tests_run),
        reviews_run=reviews_run,
        reviews_approved=reviews_approved,
        review_acceptance_rate=_rate(reviews_approved, reviews_run),
        readiness_scores_recorded=len(readiness_scores),
        average_readiness_score=round(sum(readiness_scores) / len(readiness_scores), 1) if readiness_scores else None,
    )


def compute_mttr_seconds(incidents: list) -> float | None:
    """Mean seconds from deploy_started_at (stamped at the top of graph.py's deploy(),
    i.e. right after the human approval gate, not before it — a person taking an hour to
    click approve shouldn't inflate this number) to created_at (roughly "the rollback
    gate finished resolving"). Only over incidents that actually have deploy_started_at
    set — an incident recorded before this column existed has it as None and is excluded
    outright, not treated as a 0-second recovery."""
    durations = [
        (i.created_at - i.deploy_started_at).total_seconds() for i in incidents if i.deploy_started_at is not None
    ]
    return round(sum(durations) / len(durations), 1) if durations else None


def compute_deployment_metrics(records: list, incidents: list | None = None) -> DeploymentMetrics:
    attempted = succeeded = rollbacks = 0
    strategy_breakdown: dict[str, dict[str, int]] = {}
    for r in records:
        if r.outcome not in DEPLOY_ATTEMPTED_OUTCOMES:
            continue
        attempted += 1
        strategy = r.deployment_strategy or "unknown"
        bucket = strategy_breakdown.setdefault(strategy, {"attempted": 0, "succeeded": 0, "rolled_back": 0})
        bucket["attempted"] += 1
        if r.outcome in SUCCESS_OUTCOMES:
            succeeded += 1
            bucket["succeeded"] += 1
        elif r.outcome in ROLLED_BACK_OUTCOMES:
            rollbacks += 1
            bucket["rolled_back"] += 1
    return DeploymentMetrics(
        deployments_attempted=attempted,
        deployments_succeeded=succeeded,
        deployment_success_rate=_rate(succeeded, attempted),
        rollbacks=rollbacks,
        rollback_rate=_rate(rollbacks, attempted),
        mean_time_to_recovery_seconds=compute_mttr_seconds(incidents or []),
        strategy_breakdown=strategy_breakdown,
    )


async def evaluate_platform(workspace_path: str | None = None) -> dict:
    """The I/O half: fetches real history, then hands it to the pure compute_* functions
    above. workspace_path=None evaluates across every repo this tool has ever worked on."""
    records = await memory.all_releases(workspace_path)
    incidents = await memory.all_incidents(workspace_path)
    return {
        "records_evaluated": len(records),
        "agent": compute_agent_metrics(records),
        "coding": compute_coding_metrics(records),
        "deployment": compute_deployment_metrics(records, incidents),
    }
