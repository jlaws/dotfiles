# Memory Profiling and Performance Anti-Patterns

## Common Leak Patterns

| Pattern | Language | Fix |
|---------|----------|-----|
| Event listener not removed | JS/TS | `removeEventListener` or `AbortController` |
| Closures capturing large scope | JS/TS/Python | Narrow closure scope, use `WeakRef` |
| Global caches without eviction | All | LRU cache with max size |
| Circular references + `__del__` | Python | `weakref`, avoid `__del__` |
| Goroutine leak (blocked channel) | Go | Context cancellation, buffered channels |
| Unbounded Vec/HashMap growth | Rust | `.shrink_to_fit()`, bounded collections |

## Micro-benchmark Pitfalls

| Pitfall | Problem | Mitigation |
|---------|---------|------------|
| Dead code elimination | Compiler optimizes away benchmark | `std::hint::black_box` (Rust), consume results |
| Constant folding | Inputs known at compile time | Use runtime-varying inputs |
| Cold cache vs warm cache | Results vary 10-100x | Explicitly warmup OR flush caches |
| GC pauses | Spikes in latency | Report percentiles, force GC between runs |
| Insufficient samples | High variance | Run until coefficient of variation < 5% |

## Common Performance Anti-Patterns

### Python
- String concatenation in loops (use `"".join()`)
- Global imports of heavy modules at module level (lazy import for CLI tools)
- `pandas.apply()` row-by-row (vectorize or use `numpy`)
- Synchronous I/O in async context

### JavaScript / TypeScript
- Re-renders from unstable references (`useMemo`, `useCallback`)
- Synchronous `JSON.parse` on large payloads (streaming parser)
- Unbatched DOM mutations (use `DocumentFragment` or framework batching)
- `Array.find()` in hot loops (use `Map`/`Set` for lookups)

### Go
- Excessive small allocations in hot path (pre-allocate slices, sync.Pool)
- String concatenation in loops (use `strings.Builder`)
- Unbuffered channels as queues (buffer appropriately)
- Reflection in hot paths (`reflect` is slow -- generate code instead)

### Rust
- Unnecessary cloning (borrow instead)
- `Vec` reallocation (pre-allocate with `Vec::with_capacity`)
- `Box<dyn Trait>` in hot path (monomorphize with generics)
- Debug build benchmarks (always `--release`)

## CI Regression Detection

### GitHub Actions Example

```yaml
- name: Run benchmarks
  run: |
    cargo bench -- --save-baseline current

- name: Compare with main
  run: |
    git stash
    git checkout main
    cargo bench -- --save-baseline main
    git checkout -
    git stash pop
    critcmp main current --threshold 10
```

### Regression Detection Strategies

| Approach | Pros | Cons |
|----------|------|------|
| Compare against fixed baseline | Deterministic, simple | Baseline drifts, needs periodic updates |
| Compare against main branch | Always up-to-date | CI variance can cause false positives |
| Statistical comparison (criterion) | Handles variance properly | Slower (needs many samples) |
| Performance budgets (Lighthouse CI) | Clear pass/fail | Frontend-specific |

**Recommended**: Compare against main with 5-10% tolerance to absorb CI noise.
