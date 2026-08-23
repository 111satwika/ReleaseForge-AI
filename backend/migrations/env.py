import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Registers every table on Base.metadata (ProjectMemory, ReleaseRecord, IncidentRecord,
# DeploymentState, AuditLog) — same import app.memory.db.init_db() already does for
# create_all(). This is the actual point of wiring Alembic in at all: autogenerate needs
# real model metadata to diff against, not a hand-maintained schema description that
# could drift from what the SQLAlchemy models actually say.
from app.memory.db import DB_PATH, Base
from app.memory import models  # noqa: F401  (import side effect: registers tables)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Same RELEASEFORGE_DB_PATH-aware path app.memory.db already resolves — one source of
# truth for which database file this points at, not a second hardcoded path in
# alembic.ini that could silently point somewhere else.
config.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{DB_PATH}")

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
