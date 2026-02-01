from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypedDict

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import engine
from app.db.seed import seed_if_empty
from app.graphql.schema import graphql_router
from app.rest.router import router as rest_router
from app.models import Base


settings = get_settings()


class State(TypedDict):
    """Lifespan state - typed dict for type safety."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    """
    Lifespan context manager for startup/shutdown.
    Use lifespan state instead of app.state (modern pattern).
    """
    # Startup
    if settings.is_dev:
        # Auto-create tables in dev (use Alembic in prod)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Seed mock data if empty
        if await seed_if_empty():
            print("Database seeded with mock data")

    # Increase thread pool for sync dependencies if needed
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 100  # Default is 40

    yield {}

    # Shutdown
    await engine.dispose()


app = FastAPI(
    title="LLM Tools API",
    description="GraphQL + REST API for comparison",
    version="0.1.0",
    lifespan=lifespan,
    debug=settings.debug,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_dev else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount GraphQL
app.include_router(graphql_router, prefix="/graphql")

# Mount REST API (for comparison)
app.include_router(rest_router)


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
