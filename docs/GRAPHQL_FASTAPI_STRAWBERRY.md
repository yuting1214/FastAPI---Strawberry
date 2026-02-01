# Modern GraphQL API: FastAPI + Strawberry + SQLAlchemy

> Production-ready specification with `uv`, async patterns, and performance optimizations (2025)

---

## Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Package Manager | `uv` | latest |
| API Framework | FastAPI | ≥0.115 |
| GraphQL | Strawberry | ≥0.220 |
| ORM Auto-mapping | strawberry-sqlalchemy-mapper | ≥0.8 |
| ORM | SQLAlchemy | ≥2.0 |
| Async DB Driver | asyncpg (prod) / aiosqlite (dev) | latest |
| Settings | pydantic-settings | ≥2.0 |
| Migrations | Alembic | ≥1.13 |
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

### Comparison Summary

| Metric | REST | GraphQL |
|--------|------|---------|
| HTTP Requests | 6 | 1 |
| Latency (100ms RTT) | ~600ms+ | ~100ms |
| Waterfall Dependencies | Yes | No |
| Over-fetching | Yes | No |
| Payload Size | Fixed (all fields) | Minimal (requested only) |
| Mobile vs Web | Same payload | Different queries |

---

## Mock Data Plan

### Entity Relationship

```
Provider (1) ──────< (N) Tool (N) >────── (1) Category
                          │
                          ├──< (N) Parameter
                          └──< (N) Example
```

### Seed Data Strategy

| Entity | Count | Purpose |
|--------|-------|---------|
| Provider | 3 | OpenAI, Anthropic, Google |
| Category | 4 | Text Gen, Code Gen, Image Analysis, Data Extraction |
| Tool | 7 | 2-3 per category for "related tools" demo |
| Parameter | 3-4 per tool | Show nested fetching |
| Example | 1-2 per tool | Show nested fetching |

### Mock Data Script Location

```
src/app/
└── db/
    ├── __init__.py
    └── seed.py          # Seed data for dev environment
```

### Seed Logic (in lifespan)

```python
# In main.py lifespan
if settings.is_dev:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Seed only if empty
    await seed_if_empty()
```

### Sample Seed Data

```python
PROVIDERS = [
    {"id": 1, "name": "OpenAI", "logo_url": "https://openai.com/logo.png", "website": "https://openai.com"},
    {"id": 2, "name": "Anthropic", "logo_url": "https://anthropic.com/logo.png", "website": "https://anthropic.com"},
    {"id": 3, "name": "Google", "logo_url": "https://google.com/logo.png", "website": "https://google.com"},
]

CATEGORIES = [
    {"id": 1, "name": "Text Generation", "description": "Tools for generating text content"},
    {"id": 2, "name": "Code Generation", "description": "Tools for generating and analyzing code"},
    {"id": 3, "name": "Image Analysis", "description": "Tools for analyzing and describing images"},
    {"id": 4, "name": "Data Extraction", "description": "Tools for extracting structured data"},
]

TOOLS = [
    # Text Generation (category_id=1)
    {"id": 1, "name": "Text Summarizer", "description": "Summarize long documents", "version": "2.1.0", "provider_id": 1, "category_id": 1, "usage_count": 15420},
    {"id": 2, "name": "Blog Writer", "description": "Generate blog posts", "version": "1.8.0", "provider_id": 1, "category_id": 1, "usage_count": 11300},
    # Code Generation (category_id=2)
    {"id": 3, "name": "Code Reviewer", "description": "Analyze code for bugs", "version": "1.5.0", "provider_id": 2, "category_id": 2, "usage_count": 8930},
    {"id": 4, "name": "SQL Generator", "description": "Generate SQL from natural language", "version": "3.0.0", "provider_id": 1, "category_id": 2, "usage_count": 12500},
    {"id": 5, "name": "Unit Test Generator", "description": "Generate unit tests", "version": "1.0.0", "provider_id": 2, "category_id": 2, "usage_count": 4500},
    # Image Analysis (category_id=3)
    {"id": 6, "name": "Image Captioner", "description": "Generate image descriptions", "version": "1.2.0", "provider_id": 3, "category_id": 3, "usage_count": 6780},
    # Data Extraction (category_id=4)
    {"id": 7, "name": "JSON Extractor", "description": "Extract JSON from text", "version": "2.0.0", "provider_id": 2, "category_id": 4, "usage_count": 9200},
]

PARAMETERS = [
    # Text Summarizer (tool_id=1)
    {"tool_id": 1, "name": "text", "param_type": "string", "required": True, "description": "The text to summarize"},
    {"tool_id": 1, "name": "max_length", "param_type": "integer", "required": False, "description": "Maximum summary length"},
    {"tool_id": 1, "name": "style", "param_type": "string", "required": False, "description": "Output style: bullet or paragraph"},
    # Code Reviewer (tool_id=3)
    {"tool_id": 3, "name": "code", "param_type": "string", "required": True, "description": "The code to review"},
    {"tool_id": 3, "name": "language", "param_type": "string", "required": True, "description": "Programming language"},
    {"tool_id": 3, "name": "focus", "param_type": "string", "required": False, "description": "Review focus: security, performance, style"},
    # SQL Generator (tool_id=4)
    {"tool_id": 4, "name": "query", "param_type": "string", "required": True, "description": "Natural language query description"},
    {"tool_id": 4, "name": "dialect", "param_type": "string", "required": False, "description": "SQL dialect: postgres, mysql, sqlite"},
]

EXAMPLES = [
    # Text Summarizer (tool_id=1)
    {"tool_id": 1, "title": "Article Summary", "input_text": "Long article about climate change...", "output_text": "Climate change poses significant risks..."},
    {"tool_id": 1, "title": "Meeting Notes", "input_text": "Meeting discussed Q3 targets...", "output_text": "Key points: Q3 revenue up 15%..."},
    # Code Reviewer (tool_id=3)
    {"tool_id": 3, "title": "Python Review", "input_text": "def calc(x): return x*2", "output_text": "Consider adding type hints and docstring..."},
    # SQL Generator (tool_id=4)
    {"tool_id": 4, "title": "User Query", "input_text": "Find users who signed up last month", "output_text": "SELECT * FROM users WHERE created_at >= ..."},
]
```

### Demo Query to Prove the Point

After seeding, run this query to fetch everything for Tool #1 in ONE request:

```graphql
# Single request - would be 6 REST calls
query {
  tool(id: 1) {
    name
    version
    usageCount
    provider { name website }
    category { name }
    parameters { name paramType required }
    examples { title inputText outputText }
  }
}
```

Expected response:

```json
{
  "data": {
    "tool": {
      "name": "Text Summarizer",
      "version": "2.1.0",
      "usageCount": 15420,
      "provider": {
        "name": "OpenAI",
        "website": "https://openai.com"
      },
      "category": {
        "name": "Text Generation"
      },
      "parameters": [
        {"name": "text", "paramType": "string", "required": true},
        {"name": "max_length", "paramType": "integer", "required": false},
        {"name": "style", "paramType": "string", "required": false}
      ],
      "examples": [
        {"title": "Article Summary", "inputText": "Long article...", "outputText": "Climate change..."},
        {"title": "Meeting Notes", "inputText": "Meeting discussed...", "outputText": "Key points..."}
      ]
    }
  }
}
```

---

## REST API Implementation (For Comparison)

To fairly compare REST vs GraphQL, we implement both approaches in the same codebase.

### Project Structure Update

```
src/app/
├── graphql/           # GraphQL implementation
│   └── ...
├── rest/              # REST implementation
│   ├── __init__.py
│   ├── router.py      # All REST routes
│   └── schemas.py     # Pydantic response models
└── main.py            # Mounts both /graphql and /api/v1
```

### src/app/rest/schemas.py

```python
"""Pydantic models for REST API responses."""
from datetime import datetime
from pydantic import BaseModel


class ProviderResponse(BaseModel):
    id: int
    name: str
    logo_url: str
    website: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CategoryResponse(BaseModel):
    id: int
    name: str
    description: str

    model_config = {"from_attributes": True}


class ParameterResponse(BaseModel):
    id: int
    name: str
    param_type: str
    required: bool
    description: str
    default_value: str | None = None

    model_config = {"from_attributes": True}


class ExampleResponse(BaseModel):
    id: int
    title: str
    input_text: str
    output_text: str

    model_config = {"from_attributes": True}


class ToolResponse(BaseModel):
    id: int
    name: str
    description: str
    version: str
    is_active: bool
    usage_count: int
    created_at: datetime
    provider_id: int
    category_id: int

    model_config = {"from_attributes": True}


class ToolListItem(BaseModel):
    """Lighter response for list endpoints."""
    id: int
    name: str
    description: str
    version: str

    model_config = {"from_attributes": True}


# Aggregated response for "joined" approach
class ToolDetailResponse(BaseModel):
    """Full tool detail - what GraphQL returns in 1 query."""
    id: int
    name: str
    description: str
    version: str
    is_active: bool
    usage_count: int
    created_at: datetime
    provider: ProviderResponse
    category: CategoryResponse
    parameters: list[ParameterResponse]
    examples: list[ExampleResponse]
    related_tools: list[ToolListItem]

    model_config = {"from_attributes": True}
```

### src/app/rest/router.py

```python
"""
REST API Routes - Two Approaches:

1. Traditional REST: Separate endpoints (client makes 6 requests)
2. Aggregated REST: Single endpoint with joins (server does the work)
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_session
from app.models import Tool, Provider, Category, Parameter, Example
from app.rest.schemas import (
    ToolResponse,
    ToolListItem,
    ProviderResponse,
    CategoryResponse,
    ParameterResponse,
    ExampleResponse,
    ToolDetailResponse,
)

router = APIRouter(prefix="/api/v1", tags=["REST API"])


# =============================================================================
# Approach 1: Traditional REST - Separate Endpoints
# Client must make 6 requests to build Tool Detail Page
# =============================================================================

@router.get("/tools", response_model=list[ToolResponse])
async def list_tools(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """List all tools."""
    stmt = select(Tool).limit(limit).offset(offset)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/tools/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: int, session: AsyncSession = Depends(get_session)):
    """Get tool basic info - Request 1 of 6."""
    tool = await session.get(Tool, tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.get("/tools/{tool_id}/parameters", response_model=list[ParameterResponse])
async def get_tool_parameters(tool_id: int, session: AsyncSession = Depends(get_session)):
    """Get tool parameters - Request 2 of 6."""
    stmt = select(Parameter).where(Parameter.tool_id == tool_id)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/tools/{tool_id}/examples", response_model=list[ExampleResponse])
async def get_tool_examples(tool_id: int, session: AsyncSession = Depends(get_session)):
    """Get tool examples - Request 3 of 6."""
    stmt = select(Example).where(Example.tool_id == tool_id)
    result = await session.execute(stmt)
    return result.scalars().all()


@router.get("/providers/{provider_id}", response_model=ProviderResponse)
async def get_provider(provider_id: int, session: AsyncSession = Depends(get_session)):
    """Get provider info - Request 4 of 6."""
    provider = await session.get(Provider, provider_id)
    if not provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    return provider


@router.get("/categories/{category_id}", response_model=CategoryResponse)
async def get_category(category_id: int, session: AsyncSession = Depends(get_session)):
    """Get category info - Request 5 of 6."""
    category = await session.get(Category, category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    return category


@router.get("/categories/{category_id}/tools", response_model=list[ToolListItem])
async def get_tools_by_category(
    category_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get tools in category (for related tools) - Request 6 of 6."""
    stmt = select(Tool).where(Tool.category_id == category_id)
    result = await session.execute(stmt)
    return result.scalars().all()


# =============================================================================
# Approach 2: Aggregated REST - Single Endpoint with Eager Loading
# Server does the joining, client makes 1 request
# =============================================================================

@router.get("/tools/{tool_id}/detail", response_model=ToolDetailResponse)
async def get_tool_detail(tool_id: int, session: AsyncSession = Depends(get_session)):
    """
    Get complete tool detail in ONE request.
    Uses eager loading (selectinload) to fetch all relations.
    This is the "optimized REST" approach.
    """
    stmt = (
        select(Tool)
        .where(Tool.id == tool_id)
        .options(
            selectinload(Tool.provider),
            selectinload(Tool.category),
            selectinload(Tool.parameters),
            selectinload(Tool.examples),
        )
    )
    result = await session.execute(stmt)
    tool = result.scalar_one_or_none()

    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    # Fetch related tools (same category, excluding self)
    related_stmt = (
        select(Tool)
        .where(Tool.category_id == tool.category_id)
        .where(Tool.id != tool.id)
        .limit(5)
    )
    related_result = await session.execute(related_stmt)
    related_tools = related_result.scalars().all()

    return ToolDetailResponse(
        id=tool.id,
        name=tool.name,
        description=tool.description,
        version=tool.version,
        is_active=tool.is_active,
        usage_count=tool.usage_count,
        created_at=tool.created_at,
        provider=tool.provider,
        category=tool.category,
        parameters=tool.parameters,
        examples=tool.examples,
        related_tools=related_tools,
    )
```

### Update main.py to Mount REST Router

```python
# In main.py, add:
from app.rest.router import router as rest_router

# After graphql_router
app.include_router(rest_router)
```

---

## Benchmark Comparison Script

### tests/benchmark.py

```python
"""
Benchmark: REST (sequential) vs REST (joined) vs GraphQL

Measures:
- Latency (total time)
- Number of HTTP requests
- Payload size (bytes transferred)
- Database queries executed
"""
import asyncio
import time
import json
from dataclasses import dataclass

import httpx


BASE_URL = "http://localhost:8000"
TOOL_ID = 1
ITERATIONS = 50


@dataclass
class BenchmarkResult:
    approach: str
    requests: int
    avg_latency_ms: float
    total_bytes: int
    description: str


# =============================================================================
# REST Sequential: 6 requests (simulates typical frontend)
# =============================================================================
async def benchmark_rest_sequential(client: httpx.AsyncClient) -> tuple[dict, int, int]:
    """
    Fetch tool detail using 6 separate REST calls.
    This is how a frontend typically consumes REST APIs.
    """
    total_bytes = 0
    requests = 0

    # Request 1: Tool basic info
    r1 = await client.get(f"/api/v1/tools/{TOOL_ID}")
    tool = r1.json()
    total_bytes += len(r1.content)
    requests += 1

    # Request 2: Parameters
    r2 = await client.get(f"/api/v1/tools/{TOOL_ID}/parameters")
    parameters = r2.json()
    total_bytes += len(r2.content)
    requests += 1

    # Request 3: Examples
    r3 = await client.get(f"/api/v1/tools/{TOOL_ID}/examples")
    examples = r3.json()
    total_bytes += len(r3.content)
    requests += 1

    # Request 4: Provider (depends on tool.provider_id)
    r4 = await client.get(f"/api/v1/providers/{tool['provider_id']}")
    provider = r4.json()
    total_bytes += len(r4.content)
    requests += 1

    # Request 5: Category (depends on tool.category_id)
    r5 = await client.get(f"/api/v1/categories/{tool['category_id']}")
    category = r5.json()
    total_bytes += len(r5.content)
    requests += 1

    # Request 6: Related tools (depends on category_id)
    r6 = await client.get(f"/api/v1/categories/{tool['category_id']}/tools")
    related = [t for t in r6.json() if t["id"] != TOOL_ID][:5]
    total_bytes += len(r6.content)
    requests += 1

    data = {
        "tool": tool,
        "provider": provider,
        "category": category,
        "parameters": parameters,
        "examples": examples,
        "related_tools": related,
    }
    return data, total_bytes, requests


# =============================================================================
# REST Joined: 1 request (optimized endpoint)
# =============================================================================
async def benchmark_rest_joined(client: httpx.AsyncClient) -> tuple[dict, int, int]:
    """
    Fetch tool detail using single aggregated endpoint.
    Server does the joining with eager loading.
    """
    r = await client.get(f"/api/v1/tools/{TOOL_ID}/detail")
    return r.json(), len(r.content), 1


# =============================================================================
# GraphQL: 1 request
# =============================================================================
GRAPHQL_QUERY = """
query ToolDetail($id: Int!) {
  tool(id: $id) {
    id
    name
    description
    version
    isActive
    usageCount
    createdAt
    provider {
      id
      name
      logoUrl
      website
    }
    category {
      id
      name
      description
    }
    parameters {
      id
      name
      paramType
      required
      description
      defaultValue
    }
    examples {
      id
      title
      inputText
      outputText
    }
    relatedTools {
      id
      name
      version
    }
  }
}
"""


async def benchmark_graphql(client: httpx.AsyncClient) -> tuple[dict, int, int]:
    """
    Fetch tool detail using single GraphQL query.
    """
    r = await client.post(
        "/graphql",
        json={"query": GRAPHQL_QUERY, "variables": {"id": TOOL_ID}},
    )
    return r.json(), len(r.content), 1


# =============================================================================
# Run Benchmarks
# =============================================================================
async def run_benchmark(
    name: str,
    func,
    client: httpx.AsyncClient,
    iterations: int,
) -> BenchmarkResult:
    """Run a benchmark function multiple times and collect stats."""
    latencies = []
    total_bytes = 0
    total_requests = 0

    for _ in range(iterations):
        start = time.perf_counter()
        _, bytes_transferred, requests = await func(client)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)
        total_bytes += bytes_transferred
        total_requests += requests

    return BenchmarkResult(
        approach=name,
        requests=total_requests // iterations,
        avg_latency_ms=sum(latencies) / len(latencies),
        total_bytes=total_bytes // iterations,
        description=f"Averaged over {iterations} iterations",
    )


async def main():
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: REST vs GraphQL - Tool Detail Page")
    print(f"  Tool ID: {TOOL_ID} | Iterations: {ITERATIONS}")
    print(f"{'='*70}\n")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Warmup
        await benchmark_rest_sequential(client)
        await benchmark_rest_joined(client)
        await benchmark_graphql(client)

        # Run benchmarks
        rest_seq = await run_benchmark(
            "REST Sequential (6 requests)",
            benchmark_rest_sequential,
            client,
            ITERATIONS,
        )
        rest_joined = await run_benchmark(
            "REST Joined (1 request)",
            benchmark_rest_joined,
            client,
            ITERATIONS,
        )
        graphql = await run_benchmark(
            "GraphQL (1 request)",
            benchmark_graphql,
            client,
            ITERATIONS,
        )

    # Results
    results = [rest_seq, rest_joined, graphql]

    print(f"{'Approach':<30} {'Requests':>10} {'Latency (ms)':>15} {'Payload (bytes)':>18}")
    print("-" * 75)
    for r in results:
        print(f"{r.approach:<30} {r.requests:>10} {r.avg_latency_ms:>15.2f} {r.total_bytes:>18}")

    print(f"\n{'='*70}")
    print("  ANALYSIS")
    print(f"{'='*70}")

    baseline = rest_seq.avg_latency_ms
    print(f"\n  REST Sequential (baseline): {baseline:.2f}ms")
    print(f"  REST Joined: {rest_joined.avg_latency_ms:.2f}ms ({(1 - rest_joined.avg_latency_ms/baseline)*100:.1f}% faster)")
    print(f"  GraphQL:     {graphql.avg_latency_ms:.2f}ms ({(1 - graphql.avg_latency_ms/baseline)*100:.1f}% faster)")

    if graphql.total_bytes < rest_joined.total_bytes:
        savings = (1 - graphql.total_bytes / rest_joined.total_bytes) * 100
        print(f"\n  GraphQL payload is {savings:.1f}% smaller than REST Joined")
    else:
        overhead = (graphql.total_bytes / rest_joined.total_bytes - 1) * 100
        print(f"\n  GraphQL payload has {overhead:.1f}% overhead vs REST Joined (due to field names)")

    print(f"\n  Note: GraphQL's advantage grows with:")
    print(f"    - Higher network latency (reduces round trips)")
    print(f"    - Mobile clients (smaller payloads)")
    print(f"    - Varying client needs (request only needed fields)")


if __name__ == "__main__":
    asyncio.run(main())
```

### Run Benchmark

```bash
# Terminal 1: Start server
uv run uvicorn app.main:app --port 8000

# Terminal 2: Run benchmark
uv run python tests/benchmark.py
```

### Expected Output

```
======================================================================
  BENCHMARK: REST vs GraphQL - Tool Detail Page
  Tool ID: 1 | Iterations: 50
======================================================================

Approach                          Requests    Latency (ms)    Payload (bytes)
---------------------------------------------------------------------------
REST Sequential (6 requests)             6           45.23              2847
REST Joined (1 request)                  1           12.34              1856
GraphQL (1 request)                      1           11.89              1245

======================================================================
  ANALYSIS
======================================================================

  REST Sequential (baseline): 45.23ms
  REST Joined: 12.34ms (72.7% faster)
  GraphQL:     11.89ms (73.7% faster)

  GraphQL payload is 32.9% smaller than REST Joined

  Note: GraphQL's advantage grows with:
    - Higher network latency (reduces round trips)
    - Mobile clients (smaller payloads)
    - Varying client needs (request only needed fields)
```

---

## Comparison Summary

| Metric | REST Sequential | REST Joined | GraphQL |
|--------|-----------------|-------------|---------|
| HTTP Requests | 6 | 1 | 1 |
| Waterfall Deps | Yes | No | No |
| Over-fetching | Yes | Yes | **No** |
| Payload Size | Largest | Medium | **Smallest** |
| Client Complexity | High | Low | Low |
| Flexible Fields | No | No | **Yes** |
| New Endpoint per View | Yes | Yes | **No** |
| N+1 Query Risk | Per request | Eager load | DataLoader |

### When REST Joined is Enough

- Fixed client requirements (one frontend)
- Simple data shapes
- Strong caching needs (HTTP cache-friendly)
- Team unfamiliar with GraphQL

### When GraphQL Wins

- Multiple clients (web, mobile, TV) with different data needs
- Rapidly evolving frontend requirements
- Complex nested data relationships
- Need to minimize payload for mobile
- Want single flexible endpoint vs N custom endpoints

---

## Project Structure

```
project/
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py                 # FastAPI entry + lifespan
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # Pydantic Settings (env-based)
│       │   └── database.py         # Async engine + session factory
│       ├── db/
│       │   ├── __init__.py
│       │   └── seed.py             # Mock data seeding (dev only)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py             # DeclarativeBase
│       │   ├── tool.py
│       │   ├── provider.py
│       │   └── category.py
│       ├── graphql/                # GraphQL implementation
│       │   ├── __init__.py
│       │   ├── schema.py           # Strawberry schema + context
│       │   ├── types/
│       │   │   ├── __init__.py
│       │   │   └── tool.py         # Mapped types
│       │   ├── queries.py
│       │   └── mutations.py
│       └── rest/                   # REST implementation (for comparison)
│           ├── __init__.py
│           ├── router.py           # REST routes
│           └── schemas.py          # Pydantic response models
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_graphql.py
│   └── benchmark.py                # REST vs GraphQL benchmark
├── alembic/
│   ├── env.py
│   └── versions/
├── alembic.ini
├── pyproject.toml
├── .env                            # Local dev (gitignored)
├── .env.example
└── Dockerfile
```

---

## Setup with `uv`

### Initialize Project

```bash
# Create project
uv init my-graphql-api
cd my-graphql-api

# Create src layout
mkdir -p src/app/{core,models,graphql/types} tests alembic/versions
```

### pyproject.toml

```toml
[project]
name = "my-graphql-api"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "strawberry-graphql[fastapi]>=0.220.0",
    "strawberry-sqlalchemy-mapper>=0.8.0",
    "sqlalchemy[asyncio]>=2.0.0",
    "pydantic-settings>=2.0.0",
    "alembic>=1.13.0",
    # Async drivers
    "asyncpg>=0.29.0",
    "aiosqlite>=0.19.0",
    # Performance (uvloop not on Windows)
    "uvicorn[standard]>=0.30.0",
    "uvloop>=0.19.0; sys_platform != 'win32'",
    "httptools>=0.6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "anyio>=4.0.0",
    "httpx>=0.27.0",
    "asgi-lifespan>=2.1.0",
    "ruff>=0.4.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/app"]
```

### Install Dependencies

```bash
uv sync
uv sync --group dev  # Include dev deps
```

---

## Environment Configuration

### .env.example

```bash
# App
APP_ENV=development  # development | production
DEBUG=true

# Database
# Dev: SQLite (auto-created)
# Prod: PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/dbname

# Optional overrides
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
```

### src/app/core/config.py

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_env: str = "development"
    debug: bool = False

    # Database
    database_url: str = ""

    # Pool settings (production)
    db_pool_size: int = 5
    db_max_overflow: int = 10

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"

    @property
    def db_url(self) -> str:
        """Return SQLite for dev if DATABASE_URL not set, else use DATABASE_URL."""
        if self.is_dev and not self.database_url:
            return "sqlite+aiosqlite:///./dev.db"
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## Database Setup

### src/app/core/database.py

```python
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

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
```

---

## Mock Data Seeding

### src/app/db/seed.py

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models import Provider, Category, Tool, Parameter, Example


PROVIDERS = [
    {"id": 1, "name": "OpenAI", "logo_url": "https://openai.com/logo.png", "website": "https://openai.com"},
    {"id": 2, "name": "Anthropic", "logo_url": "https://anthropic.com/logo.png", "website": "https://anthropic.com"},
    {"id": 3, "name": "Google", "logo_url": "https://google.com/logo.png", "website": "https://google.com"},
]

CATEGORIES = [
    {"id": 1, "name": "Text Generation", "description": "Tools for generating text content"},
    {"id": 2, "name": "Code Generation", "description": "Tools for generating and analyzing code"},
    {"id": 3, "name": "Image Analysis", "description": "Tools for analyzing and describing images"},
    {"id": 4, "name": "Data Extraction", "description": "Tools for extracting structured data"},
]

TOOLS = [
    {"id": 1, "name": "Text Summarizer", "description": "Summarize long documents into concise summaries", "version": "2.1.0", "provider_id": 1, "category_id": 1, "usage_count": 15420},
    {"id": 2, "name": "Blog Writer", "description": "Generate blog posts on any topic", "version": "1.8.0", "provider_id": 1, "category_id": 1, "usage_count": 11300},
    {"id": 3, "name": "Code Reviewer", "description": "Analyze code for bugs and improvements", "version": "1.5.0", "provider_id": 2, "category_id": 2, "usage_count": 8930},
    {"id": 4, "name": "SQL Generator", "description": "Generate SQL queries from natural language", "version": "3.0.0", "provider_id": 1, "category_id": 2, "usage_count": 12500},
    {"id": 5, "name": "Unit Test Generator", "description": "Generate unit tests for code", "version": "1.0.0", "provider_id": 2, "category_id": 2, "usage_count": 4500},
    {"id": 6, "name": "Image Captioner", "description": "Generate descriptions for images", "version": "1.2.0", "provider_id": 3, "category_id": 3, "usage_count": 6780},
    {"id": 7, "name": "JSON Extractor", "description": "Extract structured JSON from unstructured text", "version": "2.0.0", "provider_id": 2, "category_id": 4, "usage_count": 9200},
]

PARAMETERS = [
    # Text Summarizer
    {"tool_id": 1, "name": "text", "param_type": "string", "required": True, "description": "The text to summarize"},
    {"tool_id": 1, "name": "max_length", "param_type": "integer", "required": False, "description": "Maximum summary length in words"},
    {"tool_id": 1, "name": "style", "param_type": "string", "required": False, "description": "Output style: bullet or paragraph"},
    # Code Reviewer
    {"tool_id": 3, "name": "code", "param_type": "string", "required": True, "description": "The code to review"},
    {"tool_id": 3, "name": "language", "param_type": "string", "required": True, "description": "Programming language"},
    {"tool_id": 3, "name": "focus", "param_type": "string", "required": False, "description": "Review focus: security, performance, style"},
    # SQL Generator
    {"tool_id": 4, "name": "query", "param_type": "string", "required": True, "description": "Natural language query description"},
    {"tool_id": 4, "name": "dialect", "param_type": "string", "required": False, "description": "SQL dialect: postgres, mysql, sqlite"},
    {"tool_id": 4, "name": "schema", "param_type": "string", "required": False, "description": "Table schema context"},
    # JSON Extractor
    {"tool_id": 7, "name": "text", "param_type": "string", "required": True, "description": "Text to extract from"},
    {"tool_id": 7, "name": "schema", "param_type": "string", "required": True, "description": "Expected JSON schema"},
]

EXAMPLES = [
    # Text Summarizer
    {"tool_id": 1, "title": "Article Summary", "input_text": "Long article about climate change discussing rising temperatures, melting ice caps, and policy recommendations...", "output_text": "Climate change poses significant risks including rising sea levels and extreme weather. Key actions: reduce emissions, invest in renewables."},
    {"tool_id": 1, "title": "Meeting Notes", "input_text": "Meeting discussed Q3 targets, marketing budget allocation, and new hire onboarding process...", "output_text": "Key points: Q3 revenue target $2M, marketing budget increased 15%, 3 new hires starting next month."},
    # Code Reviewer
    {"tool_id": 3, "title": "Python Function Review", "input_text": "def calc(x): return x*2", "output_text": "Issues: 1) Add type hints (def calc(x: int) -> int), 2) Add docstring, 3) Consider more descriptive function name."},
    # SQL Generator
    {"tool_id": 4, "title": "User Query", "input_text": "Find all users who signed up last month and have made at least one purchase", "output_text": "SELECT u.* FROM users u JOIN orders o ON u.id = o.user_id WHERE u.created_at >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') GROUP BY u.id HAVING COUNT(o.id) >= 1;"},
    {"tool_id": 4, "title": "Aggregation Query", "input_text": "Count orders by status for this year", "output_text": "SELECT status, COUNT(*) as count FROM orders WHERE created_at >= DATE_TRUNC('year', CURRENT_DATE) GROUP BY status ORDER BY count DESC;"},
]


async def seed_if_empty() -> bool:
    """
    Seed database with mock data if empty.
    Returns True if seeding occurred, False if data already exists.
    """
    async with AsyncSessionLocal() as session:
        # Check if data exists
        result = await session.execute(select(Provider).limit(1))
        if result.scalar():
            return False  # Already seeded

        # Seed in order (respecting foreign keys)
        session.add_all([Provider(**p) for p in PROVIDERS])
        session.add_all([Category(**c) for c in CATEGORIES])
        await session.flush()  # Ensure IDs are available

        session.add_all([Tool(**t) for t in TOOLS])
        await session.flush()

        session.add_all([Parameter(**p) for p in PARAMETERS])
        session.add_all([Example(**e) for e in EXAMPLES])

        await session.commit()
        return True

---

## SQLAlchemy Models (Source of Truth)

### src/app/models/base.py

```python
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

### src/app/models/provider.py

```python
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.tool import Tool


class Provider(Base):
    __tablename__ = "provider"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    logo_url: Mapped[str] = mapped_column(String(500))
    website: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    tools: Mapped[list["Tool"]] = relationship(back_populates="provider")
```

### src/app/models/category.py

```python
from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.tool import Tool


class Category(Base):
    __tablename__ = "category"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)

    # Relationships
    tools: Mapped[list["Tool"]] = relationship(back_populates="category")
```

### src/app/models/tool.py

```python
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String, Text, Boolean, Integer, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.provider import Provider
    from app.models.category import Category


class Tool(Base):
    __tablename__ = "tool"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[str] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Foreign keys
    provider_id: Mapped[int] = mapped_column(ForeignKey("provider.id"))
    category_id: Mapped[int] = mapped_column(ForeignKey("category.id"))

    # Relationships → auto-mapped to GraphQL nested types
    provider: Mapped["Provider"] = relationship(back_populates="tools")
    category: Mapped["Category"] = relationship(back_populates="tools")
    parameters: Mapped[list["Parameter"]] = relationship(
        back_populates="tool", cascade="all, delete-orphan"
    )
    examples: Mapped[list["Example"]] = relationship(
        back_populates="tool", cascade="all, delete-orphan"
    )


class Parameter(Base):
    __tablename__ = "parameter"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tool.id"))
    name: Mapped[str] = mapped_column(String(100))
    param_type: Mapped[str] = mapped_column(String(50))
    required: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(Text)
    default_value: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    tool: Mapped["Tool"] = relationship(back_populates="parameters")


class Example(Base):
    __tablename__ = "example"

    id: Mapped[int] = mapped_column(primary_key=True)
    tool_id: Mapped[int] = mapped_column(ForeignKey("tool.id"))
    title: Mapped[str] = mapped_column(String(200))
    input_text: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text)

    tool: Mapped["Tool"] = relationship(back_populates="examples")
```

### src/app/models/__init__.py

```python
from app.models.base import Base
from app.models.tool import Tool, Parameter, Example
from app.models.provider import Provider
from app.models.category import Category

__all__ = ["Base", "Tool", "Parameter", "Example", "Provider", "Category"]
```

---

## GraphQL Schema with Auto-mapping

### src/app/graphql/types/__init__.py

```python
import strawberry
from sqlalchemy import select
from strawberry_sqlalchemy_mapper import StrawberrySQLAlchemyMapper

from app import models

# Single mapper instance for the entire app
mapper = StrawberrySQLAlchemyMapper()


@mapper.type(models.Provider)
class Provider:
    pass


@mapper.type(models.Category)
class Category:
    pass


@mapper.type(models.Parameter)
class Parameter:
    __exclude__ = ["tool"]  # Avoid circular back-reference


@mapper.type(models.Example)
class Example:
    __exclude__ = ["tool"]  # Avoid circular back-reference


@mapper.type(models.Tool)
class Tool:
    # Use plain lists instead of Relay connections
    __use_list__ = ["parameters", "examples"]

    @strawberry.field
    async def related_tools(self, info) -> list["Tool"]:
        """Custom resolver: Get other tools in the same category."""
        session = info.context["session"]
        stmt = (
            select(models.Tool)
            .where(models.Tool.category_id == self._category_id)
            .where(models.Tool.id != self.id)
            .limit(5)
        )
        result = await session.execute(stmt)
        return result.scalars().all()


# IMPORTANT: Finalize after all types decorated
mapper.finalize()
```

### src/app/graphql/queries.py

```python
import strawberry
from sqlalchemy import select
from strawberry.types import Info

from app import models
from app.graphql.types import Tool, Provider, Category


@strawberry.type
class Query:
    @strawberry.field
    async def tool(self, info: Info, id: int) -> Tool | None:
        session = info.context["session"]
        return await session.get(models.Tool, id)

    @strawberry.field
    async def tools(self, info: Info, limit: int = 20, offset: int = 0) -> list[Tool]:
        session = info.context["session"]
        stmt = select(models.Tool).limit(limit).offset(offset)
        result = await session.execute(stmt)
        return result.scalars().all()

    @strawberry.field
    async def providers(self, info: Info) -> list[Provider]:
        session = info.context["session"]
        result = await session.execute(select(models.Provider))
        return result.scalars().all()

    @strawberry.field
    async def categories(self, info: Info) -> list[Category]:
        session = info.context["session"]
        result = await session.execute(select(models.Category))
        return result.scalars().all()
```

### src/app/graphql/mutations.py

```python
import strawberry
from strawberry.types import Info

from app import models
from app.graphql.types import Tool


@strawberry.input
class CreateToolInput:
    name: str
    description: str
    version: str
    provider_id: int
    category_id: int


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_tool(self, info: Info, input: CreateToolInput) -> Tool:
        session = info.context["session"]
        tool = models.Tool(**strawberry.asdict(input))
        session.add(tool)
        await session.commit()
        await session.refresh(tool)
        return tool

    @strawberry.mutation
    async def delete_tool(self, info: Info, id: int) -> bool:
        session = info.context["session"]
        tool = await session.get(models.Tool, id)
        if not tool:
            return False
        await session.delete(tool)
        await session.commit()
        return True
```

### src/app/graphql/schema.py

```python
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry_sqlalchemy_mapper import StrawberrySQLAlchemyLoader

from app.core.database import AsyncSessionLocal
from app.graphql.types import mapper
from app.graphql.queries import Query
from app.graphql.mutations import Mutation


async def get_context():
    """
    Provide session and dataloader to all resolvers.
    StrawberrySQLAlchemyLoader auto-batches relationship queries (N+1 prevention).
    """
    async with AsyncSessionLocal() as session:
        yield {
            "session": session,
            "sqlalchemy_loader": StrawberrySQLAlchemyLoader(bind=session),
        }


schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    types=list(mapper.mapped_types.values()),
)

graphql_router = GraphQLRouter(
    schema,
    context_getter=get_context,
)
```

---

## FastAPI Main with Lifespan

### src/app/main.py

```python
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
            print("✅ Database seeded with mock data")

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


app = FastAPI(
    title="LLM Tools API",
    description="GraphQL API with Strawberry + SQLAlchemy",
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


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}
```

---

## Running

### Development

```bash
# With uv (auto-uses uvloop/httptools if installed)
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or explicitly
uv run python -m uvicorn app.main:app --reload
```

### Production

```bash
# Gunicorn + Uvicorn workers (recommended)
uv run gunicorn app.main:app \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000

# Or just Uvicorn
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Performance Tips (from Kludex/fastapi-tips)

### 1. uvloop + httptools Auto-enabled

When installed, Uvicorn automatically uses them. Already in dependencies:

```toml
"uvloop>=0.19.0; sys_platform != 'win32'",
"httptools>=0.6.0",
```

### 2. Always Use Async Functions

Non-async functions run in thread pool (only 40 threads by default). Penalty applies to:
- Route handlers
- Dependencies
- Background tasks

```python
# BAD - runs in thread pool
def get_client(request: Request):
    return request.state.client

# GOOD - runs in event loop
async def get_client(request: Request):
    return request.state.client
```

### 3. Increase Thread Pool if Needed

```python
# In lifespan
limiter = anyio.to_thread.current_default_thread_limiter()
limiter.total_tokens = 100  # Default is 40
```

### 4. Use Lifespan State (Not app.state)

```python
# OLD (deprecated pattern)
app.state.db = engine

# NEW (lifespan state)
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[State]:
    yield {"db": engine}  # Accessible via request.state.db
```

### 5. Debug Async Blocking

Run with `PYTHONASYNCIODEBUG=1` to find blocking calls:

```bash
PYTHONASYNCIODEBUG=1 uv run uvicorn app.main:app
```

---

## Testing

### tests/conftest.py

```python
import pytest
from httpx import ASGITransport, AsyncClient
from asgi_lifespan import LifespanManager

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client with lifespan support."""
    async with LifespanManager(app) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app),
            base_url="http://test",
        ) as ac:
            yield ac
```

### tests/test_graphql.py

```python
import pytest


@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_graphql_tools(client):
    query = """
        query {
            tools {
                id
                name
                version
            }
        }
    """
    response = await client.post("/graphql", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "tools" in data["data"]
```

### Run Tests

```bash
uv run pytest -v
```

---

## Alembic Migrations

### Initialize

```bash
uv run alembic init alembic
```

### alembic/env.py (async version)

```python
import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.core.config import get_settings
from app.models import Base

config = context.config
settings = get_settings()

# Set URL from settings
config.set_main_option("sqlalchemy.url", settings.db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Commands

```bash
# Generate migration
uv run alembic revision --autogenerate -m "add tools table"

# Apply migrations
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

---

## Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev)
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Run
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync` |
| Run dev server | `uv run uvicorn app.main:app --reload` |
| Run tests | `uv run pytest` |
| Run benchmark | `uv run python tests/benchmark.py` |
| Generate migration | `uv run alembic revision --autogenerate -m "msg"` |
| Apply migrations | `uv run alembic upgrade head` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |

---

## References

- [Kludex/fastapi-tips](https://github.com/Kludex/fastapi-tips)
- [strawberry-sqlalchemy-mapper](https://github.com/strawberry-graphql/strawberry-sqlalchemy)
- [uv documentation](https://docs.astral.sh/uv/)
- [FastAPI Lifespan](https://fastapi.tiangolo.com/advanced/events/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
