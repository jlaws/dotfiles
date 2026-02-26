# Language-Specific Profilers

## Go

```go
import (
    "net/http"
    _ "net/http/pprof"  // Register pprof handlers
)

func main() {
    go func() { http.ListenAndServe(":6060", nil) }()
    // ...
}
```

```bash
# CPU profile (30s default)
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30
# In pprof: top, list funcName, web (SVG)

# Heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Goroutine analysis
go tool pprof http://localhost:6060/debug/pprof/goroutine

# Execution tracer (scheduler, GC, goroutine events)
curl -o trace.out http://localhost:6060/debug/pprof/trace?seconds=5
go tool trace trace.out

# Benchmarks with profiling
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof
go tool pprof cpu.prof
```

```go
// Built-in benchmarks
func BenchmarkSort(b *testing.B) {
    for i := 0; i < b.N; i++ {
        sort.Ints(data)
    }
}
```

## Rust

```bash
# Flame graphs via cargo-flamegraph
cargo install flamegraph
cargo flamegraph --bin myapp

# Criterion benchmarks (statistical)
# Cargo.toml: [dev-dependencies] criterion = { version = "0.5", features = ["html_reports"] }
cargo bench

# Memory profiling with DHAT
cargo install dhat
# Annotate code with dhat::Alloc as global allocator

# Valgrind/Cachegrind (Linux)
valgrind --tool=callgrind target/release/myapp
```

## JavaScript / TypeScript

```bash
# Node.js built-in profiler
node --prof app.js
node --prof-process isolate-*.log > processed.txt

# Chrome DevTools (attach to running Node)
node --inspect app.js
# Open chrome://inspect -> Performance tab -> Record

# Clinic.js suite
npx clinic doctor -- node app.js        # Overview
npx clinic flame -- node app.js         # Flame graph
npx clinic bubbleprof -- node app.js    # Async bottlenecks

# 0x flame graphs
npx 0x app.js
```

**Browser profiling:**
- Performance tab -> Record -> Interact -> Stop -> Analyze flame chart
- Memory tab -> Heap snapshot -> Compare snapshots for leaks
- Lighthouse -> Performance audit (LCP, FID, CLS metrics)

```javascript
// Node.js: tinybench
import { Bench } from 'tinybench';
const bench = new Bench({ time: 1000 });
bench.add('sort', () => { arr.sort(); });
await bench.run();
console.table(bench.table());
```

## Python (Extended)

### memory_profiler

```bash
pip install memory_profiler
python -m memory_profiler script.py     # Line-by-line
mprof run script.py && mprof plot       # Over time
```

### tracemalloc for Leak Detection

```python
import tracemalloc
tracemalloc.start()
# ... run code ...
snapshot = tracemalloc.take_snapshot()
for stat in snapshot.statistics('lineno')[:10]:
    print(stat)
```

### Percentile Analysis

```python
import numpy as np, time

def measure_latencies(func, n=1000, warmup=50):
    """Run func n times, return percentile breakdown."""
    for _ in range(warmup):
        func()
    latencies = []
    for _ in range(n):
        start = time.perf_counter_ns()
        func()
        latencies.append(time.perf_counter_ns() - start)
    arr = np.array(latencies) / 1e6  # ms
    return {f"p{p}": np.percentile(arr, p) for p in [50, 90, 95, 99]} | {
        "mean": np.mean(arr), "std": np.std(arr),
    }
```

### Distributed Locust

```bash
# master
locust --master --expect-workers=4 -f locustfile.py
# workers (separate machines)
locust --worker --master-host=10.0.0.1 -f locustfile.py
```
