"""Database Initialization and Session Management"""

import os
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

engine = None
async_session: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    pass


async def init_db(app_data_path: str):
    global engine, async_session

    app_data = Path(app_data_path)
    app_data.mkdir(parents=True, exist_ok=True)

    # Polymers database (shared across projects)
    polymers_db = app_data / "polymers.db"

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{polymers_db}",
        echo=False,
    )

    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create tables
    from db.models import Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    global engine
    if engine:
        await engine.dispose()
        engine = None


async def get_db() -> AsyncSession:
    if async_session is None:
        raise RuntimeError("Database not initialized")
    async with async_session() as session:
        yield session
