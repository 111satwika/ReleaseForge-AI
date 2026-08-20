from sqlalchemy import select

from app.memory.db import SessionLocal
from app.memory.models import ProjectMemory, ReleaseRecord


async def get_cached_test_command(workspace_path: str) -> str | None:
    async with SessionLocal() as session:
        record = await session.get(ProjectMemory, workspace_path)
        return record.test_command if record else None


async def remember_test_command(workspace_path: str, test_command: str) -> None:
    async with SessionLocal() as session:
        record = await session.get(ProjectMemory, workspace_path)
        if record:
            record.test_command = test_command
        else:
            session.add(ProjectMemory(workspace_path=workspace_path, test_command=test_command))
        await session.commit()


async def record_release(**fields) -> None:
    async with SessionLocal() as session:
        session.add(ReleaseRecord(**fields))
        await session.commit()


async def recent_releases(workspace_path: str, limit: int = 5) -> list[ReleaseRecord]:
    async with SessionLocal() as session:
        result = await session.execute(
            select(ReleaseRecord)
            .where(ReleaseRecord.workspace_path == workspace_path)
            .order_by(ReleaseRecord.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars())
