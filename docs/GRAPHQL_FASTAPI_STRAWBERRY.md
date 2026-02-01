# Modern GraphQL API: FastAPI + Strawberry + SQLAlchemy

> Production-ready template with `uv`, async patterns, and Railway deployment (2025)

---

## Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Package Manager | `uv` | latest |
| API Framework | FastAPI | >=0.115 |
| GraphQL | Strawberry | >=0.220 |
| ORM | SQLAlchemy | >=2.0 (async) |
| Async DB Driver | asyncpg (prod) / aiosqlite (dev) | latest |
| Settings | pydantic-settings | >=2.0 |
| Server | Uvicorn + uvloop + httptools | latest |

---

## User Story: Why GraphQL?

### Scenario: LLM Tool Detail Page

**As a** frontend developer building the LLM Tools Marketplace,
**I want to** load a complete Tool Detail Page in a single request,
**So that** I can reduce latency and simplify my data fetching logic.

### The Page Requirements

```
┌─────────────────────────────────────────────────────────────────┐
│  Tool: Text Summarizer v2.1.0              Provider: [OpenAI]   │
│  Category: Text Generation                  ⭐ 15,420 uses      │
├─────────────────────────────────────────────────────────────────┤
│  Description: Summarize long documents into concise summaries   │
├──────────────────────────────┬──────────────────────────────────┤
│  Parameters:                 │  Examples:                       │
│  ├─ text (string) *required  │  ├─ "Article Summary"            │
│  ├─ max_length (int)         │  │   Input: "Long article..."    │
│  └─ style (string)           │  │   Output: "Summary..."        │
│     default: "paragraph"     │  └─ "Meeting Notes"              │
├──────────────────────────────┴──────────────────────────────────┤
│  Related Tools in "Text Generation":                            │
│  ├─ Blog Writer v1.8.0                                          │
│  └─ Email Composer v1.2.0                                       │
└─────────────────────────────────────────────────────────────────┘
```

### REST Approach: 6 Sequential Requests

```
Request 1: GET /tools/1                     → Tool basic info
Request 2: GET /tools/1/parameters          → Parameters list
Request 3: GET /tools/1/examples            → Examples list
Request 4: GET /providers/{provider_id}     → Provider details (depends on #1)
Request 5: GET /categories/{category_id}    → Category details (depends on #1)
Request 6: GET /categories/{id}/tools       → Related tools (depends on #5)
```

**Problems:**
- 6 round trips (latency × 6)
- Waterfall dependency (#4, #5, #6 wait for #1)
- Over-fetching: each endpoint returns ALL fields
- Under-fetching: need multiple requests for related data

### GraphQL Approach: 1 Request

```graphql
query ToolDetailPage($id: Int!) {
  tool(id: $id) {
    id
    name
    description
    version
    usageCount

    provider {
      name
      logoUrl
      website
    }

    category {
      name
      description
    }

    parameters {
      name
      paramType
      required
      description
      defaultValue
    }

    examples {
      title
      inputText
      outputText
    }

    # Custom resolver: tools in same category
    relatedTools {
      id
      name
      version
    }
  }
}
```

**Benefits:**
- 1 HTTP request (vs 6)
- No over-fetching: client specifies exact fields
- No waterfall: nested fields resolve in parallel
- Typed response matches UI component props

---

## Project Structure (Implemented)

```
project/
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # FastAPI entry + lifespan
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # Mode-based settings (dev/prod)
│       │   ├── init_settings.py    # CLI argument parsing
│       │   └── database.py         # Async engine + session factory
│       ├── db/
│       │   ├── __init__.py
│       │   └── seed.py             # Scalable seeding with SeedConfig
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py             # DeclarativeBase
│       │   ├── tool.py
│       │   ├── provider.py
│       │   └── category.py
│       ├── graphql/
│       │   ├── __init__.py
│       │   ├── schema.py           # Strawberry schema + session factory
│       │   ├── types/
│       │   │   └── __init__.py     # Manual Strawberry types
│       │   ├── queries.py
│       │   └── mutations.py
│       └── rest/
│           ├── __init__.py
│           ├── router.py           # REST routes
│           └── schemas.py          # Pydantic response models
├── tests/
│   ├── conftest.py
│   ├── test_graphql.py
│   ├── benchmark.py                # REST vs GraphQL benchmark
│   └── run_benchmarks.py           # Multi-scale benchmark runner
├── pyproject.toml
├── uv.lock                         # Committed for reproducible builds
├── .env.example
├── Dockerfile
└── README.md
```

---

## Environment Configuration

### Mode-Based Settings

```python
# src/app/core/config.py
class DevSettings(Settings):
    """Development settings - uses local SQLite."""
    ENV_MODE: str = "dev"
    DEBUG: bool = True

    @property
    def async_db_url(self) -> str:
        return "sqlite+aiosqlite:///./dev.db"


class ProdSettings(Settings):
    """Production settings - uses PostgreSQL via DATABASE_URL."""
    ENV_MODE: str = "prod"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    @property
    def async_db_url(self) -> str:
        url = self.DATABASE_URL
        # Heroku uses postgres:// (deprecated), convert to postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        # Add asyncpg driver
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return url


def get_settings(env_mode: str = "dev") -> Settings:
    if env_mode == "prod":
        return ProdSettings()
    return DevSettings()
```

### CLI Argument Parsing

```python
# src/app/core/init_settings.py
import os
import argparse

# Port from environment (Railway injects PORT) or default
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

parser = argparse.ArgumentParser(description="GraphQL API Server")
parser.add_argument("--mode", choices=["dev", "prod"], default="dev")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=DEFAULT_PORT)
```

---

## Database Setup

### Async Engine with Session Factory

```python
# src/app/core/database.py
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

def create_engine():
    if settings.is_dev:
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
        pool_pre_ping=True,
    )

engine = create_engine()

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)
```

---

## GraphQL Schema with Session Factory Pattern

### Critical: Concurrent Query Support

GraphQL resolvers execute **concurrently**, but SQLAlchemy async sessions don't support concurrent operations. The solution is to provide a **session factory** instead of a shared session.

```python
# src/app/graphql/schema.py
async def get_context():
    """
    Provide session factory to resolvers.

    GraphQL resolvers execute concurrently, but SQLAlchemy async sessions
    don't support concurrent operations. Each resolver gets its own session
    via the factory to avoid conflicts.
    """
    return {
        "session_factory": AsyncSessionLocal,
    }

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
)
```

### Resolver Pattern

```python
# src/app/graphql/queries.py
@strawberry.type
class Query:
    @strawberry.field
    async def tools(self, info: Info, limit: int = 20, offset: int = 0) -> list[Tool]:
        async with info.context["session_factory"]() as session:
            stmt = (
                select(models.Tool)
                .options(
                    selectinload(models.Tool.provider),
                    selectinload(models.Tool.category),
                    selectinload(models.Tool.parameters),
                    selectinload(models.Tool.examples),
                )
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return [tool_from_model(t) for t in result.scalars().all()]
```

### Manual Strawberry Types

We use manual type definitions instead of `strawberry-sqlalchemy-mapper` due to async compatibility issues:

```python
# src/app/graphql/types/__init__.py
@strawberry.type
class Tool:
    id: int
    name: str
    description: str
    version: str
    is_active: bool
    usage_count: int
    created_at: datetime
    provider_id: int
    category_id: int

    # Private fields for lazy relationship access
    _provider: strawberry.Private[Provider | None] = None
    _category: strawberry.Private[Category | None] = None
    _parameters: strawberry.Private[list[Parameter]] = strawberry.field(default_factory=list)
    _examples: strawberry.Private[list[Example]] = strawberry.field(default_factory=list)

    @strawberry.field
    def provider(self) -> Provider | None:
        return self._provider

    @strawberry.field
    async def related_tools(self, info) -> list["Tool"]:
        """Custom resolver: Get other tools in the same category."""
        async with info.context["session_factory"]() as session:
            stmt = (
                select(models.Tool)
                .where(models.Tool.category_id == self.category_id)
                .where(models.Tool.id != self.id)
                .limit(5)
            )
            result = await session.execute(stmt)
            return [tool_from_model(t) for t in result.scalars().all()]


def tool_from_model(tool: models.Tool, include_relations: bool = True) -> Tool:
    """Convert SQLAlchemy Tool model to Strawberry Tool type."""
    from sqlalchemy.inspection import inspect

    def get_if_loaded(attr_name):
        if not include_relations:
            return None
        state = inspect(tool)
        if attr_name in state.unloaded:
            return None
        return getattr(tool, attr_name, None)

    return Tool(
        id=tool.id,
        name=tool.name,
        # ... other fields
        _provider=provider_from_model(get_if_loaded("provider")) if get_if_loaded("provider") else None,
        _parameters=[param_from_model(p) for p in (get_if_loaded("parameters") or [])],
    )
```

---

## FastAPI Main with Lifespan

```python
# src/app/main.py
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
    print(f"[{settings.ENV_MODE}] Database: {settings.async_db_url.split('@')[-1]}")

    # Increase thread pool for sync operations
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 100

    yield {}

    # Shutdown
    await engine.dispose()
    print(f"[{settings.ENV_MODE}] Server stopped")


# Allow running as module: python -m app.main --mode prod
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=settings.is_dev,
    )
```

---

## Scalable Seeding

```python
# src/app/db/seed.py
@dataclass
class SeedConfig:
    """Configuration for database seeding."""
    tool_count: int = 7                    # Number of tools to create
    params_per_tool: tuple[int, int] = (1, 5)   # Min/max parameters per tool
    examples_per_tool: tuple[int, int] = (0, 3)  # Min/max examples per tool
    batch_size: int = 1000                 # Batch size for bulk inserts


async def seed_if_empty(config: SeedConfig | None = None) -> bool:
    """Seed database with data if empty."""
    config = config or SeedConfig()

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Provider).limit(1))
        if result.scalar():
            return False  # Already seeded

        # Seed providers, categories, tools, parameters, examples
        # Uses batch inserts for scalability
        ...
```

### CLI for Large-Scale Seeding

```bash
# Seed with 10,000 tools
uv run python -m app.db.seed --tools 10000

# Custom batch size
uv run python -m app.db.seed --tools 100000 --batch-size 2000
```

---

## Benchmark Results (Apple M1, SQLite, localhost)

| Dataset | Approach | Requests | Latency | Payload |
|--------:|----------|:--------:|--------:|--------:|
| **100** | REST Sequential | 6 | 4.4ms | 2.4KB |
| | REST Joined | 1 | 2.3ms | 1.7KB |
| | GraphQL | 1 | 5.9ms | 1.4KB |
| **1,000** | REST Sequential | 6 | 5.3ms | 11.8KB |
| | REST Joined | 1 | 2.8ms | 1.9KB |
| | GraphQL | 1 | 4.8ms | 1.5KB |
| **10,000** | REST Sequential | 6 | 19.4ms | 116KB |
| | REST Joined | 1 | 7.5ms | 1.9KB |
| | GraphQL | 1 | 12.8ms | 1.4KB |
| **100,000** | REST Sequential | 6 | 120.9ms | 1.1MB |
| | REST Joined | 1 | 35.6ms | 2.2KB |
| | GraphQL | 1 | 35.8ms | 1.8KB |
| **1,000,000** | REST Sequential | 6 | **1,111ms** | **10.7MB** |
| | REST Joined | 1 | 284.7ms | 1.7KB |
| | GraphQL | 1 | 289.3ms | 1.3KB |

### Key Findings

- **REST Sequential becomes unusable at scale** (payload explodes from 2KB to 11MB)
- **GraphQL and REST Joined perform nearly identically** at scale
- **GraphQL payload is consistently 20-25% smaller**

### When to Choose What

| Scenario | Recommendation |
|----------|----------------|
| Fixed queries, low latency | REST Joined |
| Mobile / high latency | **GraphQL** (smaller payloads) |
| Multiple client types | **GraphQL** (flexible queries) |
| Rapidly changing frontend | **GraphQL** (no endpoint changes) |
| Simple CRUD, fixed schema | REST (simpler tooling) |

---

## Dockerfile

```dockerfile
# Multi-stage build for smaller image
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev, frozen)
RUN uv sync --frozen --no-dev


# Production image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/

# Set Python path for module imports
ENV PYTHONPATH="/app/src"

# Run in production mode
# PORT and DATABASE_URL provided by Railway
CMD ["python", "-m", "app.main", "--mode", "prod", "--host", "0.0.0.0"]
```

---

## Railway Deployment

Railway auto-injects `DATABASE_URL` and `PORT` environment variables.

```
┌─────────────────┐     ┌─────────────────┐
│  PostgreSQL     │────▶│  FastAPI        │
│  (Railway)      │     │  (This Repo)    │
│                 │     │                 │
│  DATABASE_URL   │     │  --mode prod    │
└─────────────────┘     └─────────────────┘
```

### URL Format Support

```bash
# Railway (internal network)
DATABASE_URL=postgresql://postgres:PASSWORD@postgres.railway.internal:5432/railway

# Heroku style (auto-converted)
DATABASE_URL=postgres://user:pass@host:5432/db

# Standard PostgreSQL
DATABASE_URL=postgresql://user:pass@host:5432/db
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync --group dev` |
| Run dev server | `uv run uvicorn app.main:app --reload` |
| Run prod mode | `uv run python -m app.main --mode prod` |
| Run tests | `uv run pytest -v` |
| Run benchmark | `uv run python tests/benchmark.py` |
| Seed large dataset | `uv run python -m app.db.seed --tools 10000` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |

---

## References

- [Strawberry GraphQL](https://strawberry.rocks/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy 2.0 Async](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [uv documentation](https://docs.astral.sh/uv/)
- [Railway Templates](https://railway.app/templates)
