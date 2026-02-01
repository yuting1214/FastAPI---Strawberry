"""
Async database engine and session factory.

Supports:
- Dev: SQLite with aiosqlite
- Prod: PostgreSQL with asyncpg
"""
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.init_settings import settings


def create_engine():
    """Create async engine based on environment."""
    connect_args = {}

    if settings.is_dev:
        # SQLite needs check_same_thread=False for async
        connect_args = {"check_same_thread": False}
        return create_async_engine(
            settings.async_db_url,
            echo=settings.DEBUG,
            connect_args=connect_args,
        )

    # Production: PostgreSQL with connection pooling
    return create_async_engine(
        settings.async_db_url,
        echo=False,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True,  # Verify connections before use
    )


engine = create_engine()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session for dependency injection."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
