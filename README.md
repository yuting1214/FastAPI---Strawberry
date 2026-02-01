# GraphQL API with FastAPI + Strawberry

A production-ready GraphQL API template with FastAPI, Strawberry GraphQL, and SQLAlchemy 2.0.

## Quick Start

```bash
# Install dependencies
uv sync --group dev

# Start development server
uv run uvicorn app.main:app --reload

# Open GraphQL Playground
open http://localhost:8000/graphql
```

## Stack

| Layer | Technology |
|-------|------------|
| Package Manager | uv |
| API Framework | FastAPI |
| GraphQL | Strawberry |
| ORM | SQLAlchemy 2.0 (async) |
| Database | PostgreSQL (prod) / SQLite (dev) |
| Settings | pydantic-settings |
| Migrations | Alembic |

## Deployment

### Environment Modes

| Mode | Database | Auto-seed | Debug | Use Case |
|------|----------|-----------|-------|----------|
| `dev` | SQLite | Yes (first run) | Yes | Local development |
| `prod` | PostgreSQL | Yes (first run) | No | Production |

Both modes auto-create tables and seed sample data on first startup. Subsequent restarts skip seeding.

### Railway.com (One-Click Deploy)

1. Fork this repository
2. Create new project on Railway
3. Add **PostgreSQL** service
4. Add **GitHub Repo** service (this repo)
5. Railway auto-detects `Dockerfile`
6. The `DATABASE_URL` is automatically injected

```
┌─────────────────┐     ┌─────────────────┐
│  PostgreSQL     │────▶│  FastAPI        │
│  (Railway)      │     │  (This Repo)    │
│                 │     │                 │
│  DATABASE_URL   │     │  --mode prod    │
└─────────────────┘     └─────────────────┘
```

---

# Technical Details

## Project Structure

```
src/app/
├── core/
│   ├── config.py        # Environment settings
│   └── database.py      # Async session factory
├── models/
│   ├── tool.py          # Tool, Parameter, Example
│   ├── provider.py      # Provider
│   └── category.py      # Category
├── graphql/
│   ├── types/           # Strawberry types
│   ├── queries.py       # Query resolvers
│   ├── mutations.py     # Mutation resolvers
│   └── schema.py        # Schema configuration
├── rest/
│   ├── schemas.py       # Pydantic models
│   └── router.py        # REST endpoints
├── db/
│   └── seed.py          # Mock data
└── main.py              # App entry point
```

## API Endpoints

### GraphQL

**Endpoint:** `POST /graphql`

```graphql
# Fetch tool with all related data in ONE request
query {
  tool(id: 1) {
    id
    name
    version
    usageCount
    provider { name website }
    category { name }
    parameters { name paramType required }
    examples { title inputText outputText }
    relatedTools { id name }
  }
}

# List all tools
query {
  tools(limit: 10) {
    id
    name
    version
  }
}

# Create a tool
mutation {
  createTool(input: {
    name: "My Tool"
    description: "Description"
    version: "1.0.0"
    providerId: 1
    categoryId: 1
  }) {
    id
    name
  }
}
```

### REST

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/tools` | List all tools |
| `GET /api/v1/tools/{id}` | Get tool basic info |
| `GET /api/v1/tools/{id}/detail` | Get tool with all relations |
| `GET /api/v1/tools/{id}/parameters` | Get tool parameters |
| `GET /api/v1/tools/{id}/examples` | Get tool examples |
| `GET /api/v1/providers/{id}` | Get provider |
| `GET /api/v1/categories/{id}` | Get category |
| `GET /api/v1/categories/{id}/tools` | Get tools by category |

## Development

### Commands

```bash
# Install dependencies
uv sync --group dev

# Run development server
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest -v

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

### Database Migrations

```bash
# Generate migration
uv run alembic revision --autogenerate -m "description"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

### Production Migrations

```bash
export APP_MODE=prod
export DATABASE_URL=postgresql://...

uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
```

## Mock Data & Seeding

### Default Sample Data

On first startup, the server seeds:
- **10 Providers** (OpenAI, Anthropic, Google, Meta, etc.)
- **12 Categories** (Text Generation, Code Generation, etc.)
- **7 Tools** with parameters and examples

### Large Scale Seeding

For performance testing:

```bash
# Delete existing database and seed with 10,000 tools
rm dev.db
uv run python -m app.db.seed --tools 10000

# Custom configuration
uv run python -m app.db.seed --tools 50000 --batch-size 2000
```

### Programmatic Seeding

```python
from app.db.seed import seed_if_empty, SeedConfig

config = SeedConfig(
    tool_count=10000,
    params_per_tool=(1, 5),
    examples_per_tool=(0, 3),
    batch_size=1000,
)
await seed_if_empty(config)
```

---

# Benchmark: REST vs GraphQL

## Run Benchmarks

```bash
# Quick benchmark (requires running server)
uv run uvicorn app.main:app --port 8000  # Terminal 1
uv run python tests/benchmark.py          # Terminal 2

# Full benchmark at multiple scales
uv run python tests/run_benchmarks.py

# 1M scale benchmark
uv run python -m app.db.seed --tools 1000000
uv run python tests/benchmark_1m.py
```

## Results (Apple M1, SQLite, localhost)

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

## Key Findings

**REST Sequential is unusable at scale:**
- Payload explodes: 2KB → 11MB (5,500x increase)
- Latency explodes: 4ms → 1,111ms (278x slower)

**GraphQL vs REST Joined:**

| Scale | Winner | Difference |
|------:|--------|------------|
| 100 | REST Joined | 61% faster |
| 1,000 | REST Joined | 41% faster |
| 10,000 | REST Joined | 41% faster |
| 100,000 | **Tie** | <1% difference |
| 1,000,000 | **Tie** | <2% difference |

**GraphQL payload is consistently 20-25% smaller** across all scales.

## When to Choose What

| Scenario | Recommendation |
|----------|----------------|
| Fixed queries, low latency | REST Joined |
| Mobile / high latency | **GraphQL** (smaller payloads) |
| Multiple client types | **GraphQL** (flexible queries) |
| Rapidly changing frontend | **GraphQL** (no endpoint changes) |
| Simple CRUD, fixed schema | REST (simpler tooling) |

## The Real Story

**Well-designed REST performs nearly identically to GraphQL** for equivalent queries. The choice is about:

1. **Flexibility**: GraphQL lets clients request exactly what they need
2. **Payload efficiency**: GraphQL is 20-25% smaller (matters for mobile)
3. **Developer experience**: One flexible endpoint vs many fixed endpoints
4. **Versioning**: GraphQL schema evolution vs REST API versioning

---

## License

MIT
