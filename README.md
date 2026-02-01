# GraphQL API with FastAPI + Strawberry

A production-ready GraphQL API template demonstrating modern async Python patterns with FastAPI, Strawberry GraphQL, and SQLAlchemy 2.0.

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

### REST (for comparison)

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

# Run benchmark (server must be running)
uv run python tests/benchmark.py

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

### Environment Variables

Copy `.env.example` to `.env`:

```bash
# App
APP_ENV=development
DEBUG=true

# Database (leave empty for SQLite in dev)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/dbname
```

## Benchmark: REST vs GraphQL

Run the benchmark to compare approaches:

```bash
# Terminal 1: Start server
uv run uvicorn app.main:app --port 8000

# Terminal 2: Run benchmark
uv run python tests/benchmark.py
```

**Sample Results:**

| Approach | Requests | Latency | Payload |
|----------|----------|---------|---------|
| REST Sequential (6 requests) | 6 | ~45ms | ~2.8KB |
| REST Joined (1 request) | 1 | ~12ms | ~1.8KB |
| GraphQL (1 request) | 1 | ~12ms | ~1.2KB |

GraphQL advantages:
- Single request vs multiple round trips
- Client specifies exact fields needed
- Smaller payload (no over-fetching)
- Flexible queries without new endpoints

## Mock Data

The development server auto-seeds with sample data on startup.

### Default (Quick Dev)

Small dataset for fast iteration:
- **10 Providers** (OpenAI, Anthropic, Google, Meta, etc.)
- **12 Categories** (Text Generation, Code Generation, etc.)
- **7 Tools** with parameters and examples

### Large Scale (10,000+ tools)

For performance testing and benchmarking:

```bash
# Delete existing database and seed with 10,000 tools
rm dev.db
uv run python -m app.db.seed --tools 10000

# Custom configuration
uv run python -m app.db.seed --tools 50000 --batch-size 2000
```

**Sample output (10K tools):**
```
Seeded: 10 providers, 12 categories, 10000 tools, 29820 parameters, 15112 examples
Completed in 8.43 seconds
```

### Programmatic Seeding

```python
from app.db.seed import seed_if_empty, SeedConfig

# Large dataset
config = SeedConfig(
    tool_count=10000,
    params_per_tool=(1, 5),      # 1-5 parameters per tool
    examples_per_tool=(0, 3),    # 0-3 examples per tool
    batch_size=1000,             # Insert batch size
)
await seed_if_empty(config)
```

## Production

### Docker

```bash
docker build -t graphql-api .
docker run -p 8000:8000 -e DATABASE_URL=postgresql+asyncpg://... graphql-api
```

### Gunicorn

```bash
uv run gunicorn app.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000
```

## License

MIT
