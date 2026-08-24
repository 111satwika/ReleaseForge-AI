"""First unit test for anything in app/agents/graph.py directly — every other node in
this file is orchestration (LLM calls, MCP calls, LangGraph state transitions) that this
project has always live-tested against real infrastructure instead of unit-tested (see
RELEASEFORGE_BUILD_PLAN.md for why). _partial_credit_line() is different: genuinely pure
logic with no I/O, added for release_readiness()'s three new real checks, and a real
scoring rule worth locking in directly."""

from app.agents.graph import _partial_credit_line


def test_partial_credit_full_points_when_ran_and_clean():
    points, line = _partial_credit_line("Security scan", ran=True, clean=True, points_if_clean=10, detail="No findings.")
    assert points == 10
    assert "(+10)" in line
    assert "No findings." in line


def test_partial_credit_zero_when_ran_and_not_clean():
    points, line = _partial_credit_line("Dependency check", ran=True, clean=False, points_if_clean=10, detail="2 vulnerabilities.")
    assert points == 0
    assert "(+0)" in line


def test_partial_credit_half_when_it_could_not_run_at_all():
    """A check that couldn't run is neither "verified clean" nor "known bad" — it should
    score strictly between those two outcomes, not collapse to either one."""
    points, line = _partial_credit_line("Infrastructure readiness", ran=False, clean=False, points_if_clean=10, detail="kubectl isn't installed.")
    assert points == 5
    assert 0 < points < 10
    assert "couldn't run" in line
    assert "kubectl isn't installed." in line
