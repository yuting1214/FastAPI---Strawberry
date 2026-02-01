#!/usr/bin/env python3
"""
Run benchmarks at multiple dataset sizes.
Automatically seeds database, starts server, runs benchmarks.
"""
import asyncio
import subprocess
import time
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import httpx

BASE_URL = "http://localhost:8000"
PORT = 8000
ITERATIONS = 20
TOOL_SIZES = [100, 1000, 10000, 100000]

GRAPHQL_QUERY = """
query ToolDetail($id: Int!) {
  tool(id: $id) {
    id name description version isActive usageCount
    provider { id name logoUrl website }
    category { id name description }
    parameters { id name paramType required description }
    examples { id title inputText outputText }
    relatedTools { id name version }
  }
}
"""


async def wait_for_server(timeout=30):
    """Wait for server to be ready."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = await client.get(f"{BASE_URL}/health")
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


async def benchmark_rest_sequential(client, tool_id):
    total_bytes = 0
    r1 = await client.get(f"/api/v1/tools/{tool_id}")
    tool = r1.json()
    total_bytes += len(r1.content)
    r2 = await client.get(f"/api/v1/tools/{tool_id}/parameters")
    total_bytes += len(r2.content)
    r3 = await client.get(f"/api/v1/tools/{tool_id}/examples")
    total_bytes += len(r3.content)
    r4 = await client.get(f"/api/v1/providers/{tool['provider_id']}")
    total_bytes += len(r4.content)
    r5 = await client.get(f"/api/v1/categories/{tool['category_id']}")
    total_bytes += len(r5.content)
    r6 = await client.get(f"/api/v1/categories/{tool['category_id']}/tools?limit=5")
    total_bytes += len(r6.content)
    return total_bytes, 6


async def benchmark_rest_joined(client, tool_id):
    r = await client.get(f"/api/v1/tools/{tool_id}/detail")
    return len(r.content), 1


async def benchmark_graphql(client, tool_id):
    r = await client.post("/graphql", json={"query": GRAPHQL_QUERY, "variables": {"id": tool_id}})
    return len(r.content), 1


async def run_benchmark(name, func, client, tool_id, iterations):
    latencies = []
    total_bytes = 0

    # Warmup
    for _ in range(3):
        await func(client, tool_id)

    for _ in range(iterations):
        start = time.perf_counter()
        bytes_transferred, _ = await func(client, tool_id)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        total_bytes += bytes_transferred

    return {
        "name": name,
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "bytes": total_bytes // iterations,
    }


async def run_benchmarks_for_size(tool_count):
    tool_id = max(1, tool_count // 2)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        rest_seq = await run_benchmark("REST Sequential (6 req)", benchmark_rest_sequential, client, tool_id, ITERATIONS)
        rest_join = await run_benchmark("REST Joined (1 req)", benchmark_rest_joined, client, tool_id, ITERATIONS)
        graphql = await run_benchmark("GraphQL (1 req)", benchmark_graphql, client, tool_id, ITERATIONS)

    return {
        "size": tool_count,
        "rest_seq": rest_seq,
        "rest_join": rest_join,
        "graphql": graphql,
    }


def seed_database(tool_count):
    """Seed database with specified number of tools."""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'dev.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    result = subprocess.run(
        ["uv", "run", "python", "-m", "app.db.seed", "--tools", str(tool_count)],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), '..'),
    )
    return "Seeded:" in result.stdout or "Seeded:" in result.stderr


def start_server():
    """Start uvicorn server."""
    return subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=os.path.join(os.path.dirname(__file__), '..'),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_server(proc):
    """Stop server."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()


async def main():
    print("=" * 70)
    print("  Benchmark: REST vs GraphQL Performance at Scale")
    print(f"  Iterations: {ITERATIONS} | Dataset sizes: {TOOL_SIZES}")
    print("=" * 70)

    all_results = []

    for tool_count in TOOL_SIZES:
        print(f"\n[{tool_count:,} tools] Seeding database...")
        if not seed_database(tool_count):
            print("  Failed to seed database!")
            continue

        print(f"[{tool_count:,} tools] Starting server...")
        server = start_server()

        try:
            if not await wait_for_server():
                print("  Server failed to start!")
                continue

            print(f"[{tool_count:,} tools] Running benchmarks...")
            result = await run_benchmarks_for_size(tool_count)
            all_results.append(result)

        finally:
            stop_server(server)
            await asyncio.sleep(1)

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print("\n| Dataset | Approach | Requests | Avg Latency | Payload |")
    print("|--------:|----------|:--------:|------------:|--------:|")

    for r in all_results:
        size = f"{r['size']:,}"
        for key in ['rest_seq', 'rest_join', 'graphql']:
            b = r[key]
            print(f"| {size} | {b['name']} | {6 if 'Sequential' in b['name'] else 1} | {b['avg_ms']:.2f}ms | {b['bytes']:,}B |")
        size = ""  # Only show size on first row

    print("\n### Performance Analysis\n")
    for r in all_results:
        seq = r['rest_seq']['avg_ms']
        join = r['rest_join']['avg_ms']
        gql = r['graphql']['avg_ms']

        join_pct = (1 - join/seq) * 100
        gql_pct = (1 - gql/seq) * 100

        payload_save = (1 - r['graphql']['bytes'] / r['rest_join']['bytes']) * 100

        print(f"**{r['size']:,} tools:**")
        print(f"- REST Joined: **{join_pct:.0f}%** faster than Sequential ({seq:.1f}ms → {join:.1f}ms)")
        print(f"- GraphQL: **{gql_pct:.0f}%** faster than Sequential ({seq:.1f}ms → {gql:.1f}ms)")
        print(f"- GraphQL payload: **{abs(payload_save):.0f}%** {'smaller' if payload_save > 0 else 'larger'} than REST Joined")
        print()


if __name__ == "__main__":
    asyncio.run(main())
