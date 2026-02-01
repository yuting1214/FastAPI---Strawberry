"""
FastAPI + Strawberry GraphQL API

Supports:
- Dev mode: SQLite, debug enabled
- Prod mode: PostgreSQL via DATABASE_URL

Both modes auto-create tables and seed sample data on first startup.
Subsequent restarts skip seeding (data already exists).

Usage:
    # Development (default)
    uvicorn app.main:app --reload

    # Production (via module)
    python -m app.main --mode prod --host 0.0.0.0 --port 8000

    # Production (via Docker)
    docker run -e DATABASE_URL=... -p 8000:8000 image
"""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypedDict

import anyio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.init_settings import settings, args
from app.core.database import engine
from app.db.seed import seed_if_empty
from app.graphql.schema import graphql_router
from app.rest.router import router as rest_router
from app.models import Base


class State(TypedDict):
    """Lifespan state."""
    pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    """Startup and shutdown logic."""
    # Startup: Create tables if they don't exist (both dev and prod)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed sample data only if database is empty (first-time setup)
    if await seed_if_empty():
        print(f"[{settings.ENV_MODE}] Database seeded with sample data")

    print(f"[{settings.ENV_MODE}] Server starting...")
    print(f"[{settings.ENV_MODE}] Database: {settings.async_db_url.split('@')[-1] if '@' in settings.async_db_url else settings.async_db_url}")

    # Increase thread pool for sync operations
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 100

    yield {}

    # Shutdown
    await engine.dispose()
    print(f"[{settings.ENV_MODE}] Server stopped")


app = FastAPI(
    title=settings.APP_NAME,
    description="GraphQL + REST API with FastAPI and Strawberry",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    debug=settings.DEBUG,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_dev else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(graphql_router, prefix="/graphql")
app.include_router(rest_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mode": settings.ENV_MODE,
        "version": settings.APP_VERSION,
    }


# Allow running as module: python -m app.main --mode prod
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=settings.is_dev,
    )
