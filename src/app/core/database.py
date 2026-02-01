from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

settings = get_settings()


def create_engine():
    """Create async engine with env-appropriate settings."""
    connect_args = {}

    if settings.is_dev:
        # SQLite: need check_same_thread for async
        if "sqlite" in settings.db_url:
            connect_args = {"check_same_thread": False}
        return create_async_engine(
            settings.db_url,
            echo=settings.debug,
            connect_args=connect_args,
        )

    # Production: PostgreSQL with connection pooling
    return create_async_engine(
        settings.db_url,
        echo=False,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
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
    """Yield an async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
