from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.memory.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ProjectMemory(Base):
    """Facts DeployGenie has learned about a repo, so it doesn't re-derive them every
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

    # e.g. "declined_at_plan", "declined_at_code", "declined_at_test",
    # "declined_at_readiness", "review_unresolved", "docker_built", "docker_build_failed"
    outcome: Mapped[str] = mapped_column(String)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
