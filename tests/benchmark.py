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
