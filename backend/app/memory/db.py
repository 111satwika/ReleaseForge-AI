from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

# SQLite for now — Postgres was the original plan, but it isn't installed on this
# machine and there's no Docker to containerize it. SQLAlchemy makes that swap later
# a connection-string change, not a rewrite: only db.py and the URL below would move.
DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}")
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    from app.memory import models  # noqa: F401  (registers tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
