from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.memory.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectMemory(Base):
    """Facts ReleaseForge AI has learned about a repo, so it doesn't re-derive them every
    task. Only test_command exists today — framework/build-command detection isn't
    built yet, so this deliberately doesn't pretend to know them."""

    __tablename__ = "project_memory"

    workspace_path: Mapped[str] = mapped_column(String, primary_key=True)
    test_command: Mapped[str | None] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class ReleaseRecord(Base):
    """One row per completed task run — however it ended. Declined-and-abandoned
    attempts are recorded too, not just successful ones; the point is a real history,
    not a highlight reel."""

    __tablename__ = "release_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_path: Mapped[str] = mapped_column(String, index=True)

    target_file: Mapped[str | None] = mapped_column(String, nullable=True)
    plan: Mapped[str | None] = mapped_column(String, nullable=True)
    change_description: Mapped[str | None] = mapped_column(String, nullable=True)

    test_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    review_status: Mapped[str | None] = mapped_column(String, nullable=True)
    review_risk: Mapped[str | None] = mapped_column(String, nullable=True)

    readiness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    readiness_recommendation: Mapped[str | None] = mapped_column(String, nullable=True)

    branch_name: Mapped[str | None] = mapped_column(String, nullable=True)
    pr_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Added in Phase 8 for deployment evaluation metrics (success/rollback rate by
    # strategy) — null for any record from before this column existed, and for records
    # that never reached deployment. No migration tooling exists yet (no Alembic), so an
    # existing data/memory.db from before Phase 8 needs a fresh database to pick this up.
    deployment_strategy: Mapped[str | None] = mapped_column(String, nullable=True)

    # e.g. "declined_at_plan", "declined_at_code", "declined_at_test",
    # "declined_at_readiness", "declined_at_deploy", "declined_at_rollback",
    # "docker_build_failed", "review_unresolved", "planning_failed", "publish_failed",
    # "deployed", "rolled_back"
    outcome: Mapped[str] = mapped_column(String)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class IncidentRecord(Base):
    """One row per rollback event — the root-cause analysis Phase 4's Recovery agent
    already produces on every failed deployment, actually persisted instead of shown
    once in a chat reply and discarded. Closes a gap Phase 5 flagged and never built:
    at the time, Phase 4's Recovery agent didn't exist yet to produce this data at all."""

    __tablename__ = "incident_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    workspace_path: Mapped[str] = mapped_column(String, index=True)
    release_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    deployment_strategy: Mapped[str | None] = mapped_column(String, nullable=True)
    # "rolled_back" | "left_as_is" — matches the two ways await_rollback_approval's gate
    # can resolve (rollback() vs. declined_at_rollback()).
    action: Mapped[str] = mapped_column(String)
    root_cause: Mapped[str] = mapped_column(String)
    deploy_log: Mapped[str | None] = mapped_column(String, nullable=True)

    # Set from AgentState["deploy_started_at"] (stamped at the top of deploy()) —
    # the piece that was actually missing for a real MTTR metric, not just a rename of
    # an existing field. `created_at` below is roughly "when the rollback finished," so
    # created_at - deploy_started_at is a real detection-to-recovery duration, not
    # inferred from two records that otherwise both get written in the same request.
    deploy_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)


class DeploymentState(Base):
    """One row per (workspace_path, app_name): the Kubernetes Deployment resource name
    actually serving stable traffic right now, tracked as real state instead of inferred
    from naming conventions. Closes the gap Phase 4's own build-plan notes flagged and
    left open — promote_and_finish()'s Canary branch used to assume the stable version
    always lives at a plain-named Deployment (`<app>`), which silently failed to promote
    into a stable version actually running under a Blue-Green name (`<app>-green`) and,
    in a real live test, deleted the canary's serving pods for a promotion that never
    happened. No history is kept here on purpose — this is "what's true right now,"
    ReleaseRecord/IncidentRecord already own the history."""

    __tablename__ = "deployment_state"

    workspace_path: Mapped[str] = mapped_column(String, primary_key=True)
    app_name: Mapped[str] = mapped_column(String, primary_key=True)
    current_deployment_name: Mapped[str] = mapped_column(String)
    current_strategy: Mapped[str] = mapped_column(String)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class AuditLog(Base):
    """One row per MCP tool call this backend ever attempted — the design doc's Section
    14 "audit logs" requirement, made real. Every attempt is recorded, not just
    successful ones: a scope-denied call (an agent role reaching for a tool outside its
    least-privilege scope) is exactly the kind of event an audit log exists to catch,
    so it's logged here too, not silently swallowed as an exception."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String, index=True)
    tool_name: Mapped[str] = mapped_column(String, index=True)
    server_name: Mapped[str | None] = mapped_column(String, nullable=True)
    workspace_path: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

    # "ok" | "error" | "scope_denied"
    status: Mapped[str] = mapped_column(String)
    detail: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
