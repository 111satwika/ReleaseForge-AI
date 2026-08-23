from sqlalchemy import select

from app.memory.db import SessionLocal
from app.memory.models import AuditLog


async def record_audit_event(
    role: str,
    tool_name: str,
    status: str,
    server_name: str | None = None,
    workspace_path: str | None = None,
    detail: str | None = None,
) -> None:
    async with SessionLocal() as session:
        session.add(
            AuditLog(
                role=role,
                tool_name=tool_name,
                server_name=server_name,
                workspace_path=workspace_path,
                status=status,
                detail=detail,
            )
        )
        await session.commit()


async def recent_audit_events(limit: int = 100, workspace_path: str | None = None) -> list[AuditLog]:
    async with SessionLocal() as session:
        query = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)
        if workspace_path is not None:
            query = query.where(AuditLog.workspace_path == workspace_path)
        result = await session.execute(query)
        return list(result.scalars())
