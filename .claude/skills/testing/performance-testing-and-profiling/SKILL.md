---
name: performance-testing-and-profiling
description: "Load testing, CPU/memory/latency profiling, benchmarking, and performance regression detection. Use when load testing APIs, profiling CPU/memory/latency, benchmarking code, or detecting performance regressions. Covers Locust/k6 load scenarios, flame graphs, language-specific profilers, and CI regression detection. Do NOT use for observability infrastructure setup (use observability) or general bug investigation (use debugging-methodology)."
---

# Performance Testing and Profiling

## The Rule

```
MEASURE FIRST. OPTIMIZE SECOND. MEASURE AGAIN.
```

Never optimize based on intuition. Profile, identify the bottleneck, fix it, verify with numbers.

## Tool Decision Table

| Scenario | Tool | Why |
|----------|------|-----|
| HTTP API load testing | Locust | Python-native, programmable scenarios, distributed |
| Protocol-level perf (gRPC, WS) | k6 | Built-in protocol support, JS scripting |
| Production CPU profiling (Python) | py-spy | No code changes, low overhead, flame graphs |
| Function-level hotspots (Python) | line_profiler | Per-line timing, surgical precision |
| Memory profiling (Python) | memray / tracemalloc | Native + Python allocations, leak detection |
| Go profiling | pprof | CPU, heap, goroutine, execution tracing |
| Rust profiling | cargo-flamegraph | Flame graphs from perf/dtrace |
| JS/Node profiling | Clinic.js / Chrome DevTools | Flame charts, async bottlenecks |
| DB query performance | EXPLAIN ANALYZE | Actual execution plan with timing |
| Microbenchmarks | pytest-benchmark / criterion / testing.B | Statistical rigor per language |

## Symptom-Based Profiling

| Symptom | Tool Category | What to Look For |
|---------|---------------|------------------|
| Slow response times | CPU profiler / flame graph | Hot functions, deep call stacks |
| High memory usage | Heap profiler / snapshots | Leaked objects, growing allocations |
| Memory keeps growing | Allocation tracker + GC logs | Unreleased references, finalizer queues |
| High latency variance | Distributed tracing | Slow downstream calls, queue backpressure |
| Event loop lag (Node) | Clinic.js / blocked-at | Sync I/O, CPU-heavy on main thread |
| Slow page load | Lighthouse | Large bundles, render-blocking resources |

## Load Testing

### Locust (Python)

```python
# locustfile.py
from locust import HttpUser, task, between, tag, LoadTestShape

class APIUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    def on_start(self):
        resp = self.client.post("/auth/login", json={
            "username": "loadtest", "password": "secret"
        })
        self.token = resp.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    @tag("read")
    @task(10)
    def list_items(self):
        with self.client.get("/api/items", headers=self.headers,
                             catch_response=True) as resp:
            if resp.status_code == 200 and len(resp.json()) > 0:
                resp.success()
            else:
                resp.failure(f"Unexpected: {resp.status_code}")

    @tag("write")
    @task(1)
    def create_item(self):
        self.client.post("/api/items", headers=self.headers,
                         json={"name": "load-test-item", "value": 42})

class StepLoadShape(LoadTestShape):
    """Ramp up in steps: 10 users every 30s up to 100, hold, ramp down."""
    stages = [
        {"duration": 30,  "users": 10,  "spawn_rate": 10},
        {"duration": 60,  "users": 50,  "spawn_rate": 10},
        {"duration": 120, "users": 100, "spawn_rate": 10},
        {"duration": 240, "users": 100, "spawn_rate": 10},  # hold
        {"duration": 270, "users": 0,   "spawn_rate": 10},  # ramp down
    ]
    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        return None
```

### k6 (JavaScript)

```javascript
// load_test.js
import http from "k6/http";
import { check, sleep } from "k6";
import { Rate } from "k6/metrics";

const errorRate = new Rate("errors");

export const options = {
  stages: [
    { duration: "30s", target: 20 },
    { duration: "1m",  target: 20 },
    { duration: "10s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<500", "p(99)<1000"],
    errors: ["rate<0.01"],
  },
};

export default function () {
  const res = http.get("http://localhost:8000/api/items");
  check(res, {
    "status is 200": (r) => r.status === 200,
    "body has items": (r) => JSON.parse(r.body).length > 0,
  }) || errorRate.add(1);
  sleep(1);
}
```

## Profiling Quick Reference

### Python

```bash
# CPU sampling (attach to running process, low overhead)
py-spy record -o profile.svg --pid $PID
py-spy record -o profile.svg -- python app.py

# Deterministic profiling
python -m cProfile -o stats.prof app.py
# Visualize: snakeviz stats.prof

# Line-level hotspots
# pip install line_profiler
kernprof -l -v script.py  # decorate target functions with @profile

# Memory leak detection
python -c "import tracemalloc; tracemalloc.start(); ..."
```

| Tool | Overhead | Use Case |
|------|----------|----------|
| py-spy | ~5% | Production sampling, flame graphs |
| cProfile | ~50-200% | Deterministic call counts, dev only |
| line_profiler | 10-50x | Targeted hot functions only |
| memray | Medium | Native + Python memory allocations |
| tracemalloc | Medium | Leak detection, allocation tracing |
| scalene | Low-Med | CPU + memory + GPU combined |

## Flame Graph Interpretation

```
Reading:
- X-axis = sample population (NOT time sequence)
- Y-axis = stack depth (bottom = entry, top = leaf)
- Width = proportion of total samples (wider = more CPU)

Look for:
1. PLATEAUS at top -- functions themselves slow (no hot children)
2. WIDE towers -- deep stacks dominating CPU
3. Unexpected functions -- framework code you didn't expect

Action:
- Wide plateau at top -> optimize that function
- Wide tower -> algorithmic change or caching higher up
- Many narrow towers -> death by 1000 cuts; batch operations
```

## Benchmarking Protocol

1. **Warmup**: Discard first N iterations (JIT, caches, OS scheduling)
2. **Samples**: >30 iterations for statistical significance
3. **Isolation**: Dedicated runners, pin CPU freq, close other apps
4. **Baseline**: Always compare against known baseline
5. **Report**: Median + p95 + p99 (not just mean)

### pytest-benchmark

```python
def test_serialization_perf(benchmark):
    data = {"users": [{"id": i, "name": f"user_{i}"} for i in range(1000)]}
    result = benchmark(json.dumps, data)
    assert result is not None
# Run: pytest test_perf.py --benchmark-only --benchmark-histogram
```

## Database Query Profiling

```sql
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT u.id, u.email, count(o.id) as order_count
FROM users u LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > now() - interval '30 days'
GROUP BY u.id;
```

```python
# find_slow_queries.py -- requires pg_stat_statements extension
def find_slow_queries(conn, min_mean_ms=100, limit=20):
    cur = conn.cursor()
    cur.execute("""
        SELECT query, calls, mean_exec_time, total_exec_time, rows
        FROM pg_stat_statements
        WHERE mean_exec_time > %s
        ORDER BY total_exec_time DESC LIMIT %s
    """, (min_mean_ms, limit))
    for query, calls, mean_ms, total_ms, rows in cur.fetchall():
        print(f"Mean: {mean_ms:.1f}ms | Calls: {calls} | "
              f"Total: {total_ms/1000:.1f}s\n  {query[:120]}\n")
```

## Gotchas

- **Locust counts RPS per worker** -- aggregate in master UI, not per-worker logs
- **py-spy needs root on Linux** -- use `--nonblocking` or run as root; macOS needs SIP consideration
- **cProfile overhead ~2x** -- never in production; use py-spy for live systems
- **k6 thresholds are assertions** -- non-zero exit if thresholds fail; wire into CI
- **Warmup matters** -- first N requests hit cold caches/JIT; always discard
- **Coordinated omission** -- if tool waits for response before sending next, tail latency is undercounted
- **EXPLAIN vs EXPLAIN ANALYZE** -- plain EXPLAIN = estimated plan; ANALYZE executes query
- **Go pprof heap** -- shows live objects by default; use `-alloc_space` for total allocations
- **Rust benchmarks MUST use `--release`** -- debug builds 10-100x slower
- **CI benchmark noise** -- use dedicated runners or statistical comparison with 5-10% tolerance
- **Memory profiler overhead** can change GC behavior, hiding/creating leaks
- **Flame graphs from short runs** are noisy; collect 10-30 seconds minimum

## Extended References

See `references/` for:
- **language-profilers.md** -- Go pprof, Rust flamegraph/criterion, JS Clinic.js, detailed Python memory profiling
- **memory-and-antipatterns.md** -- Memory leak patterns by language, common performance anti-patterns, CI regression detection

## Cross-References

- **testing:debugging-methodology** -- systematic debugging when profiling reveals unexpected behavior
