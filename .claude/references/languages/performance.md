# Profiling Strategy

| Need | Tool | Command |
|------|------|---------|
| Where is time spent? | `cProfile` | `python -m cProfile -o out.prof script.py` |
| Line-by-line timing | `line_profiler` | `kernprof -l -v script.py` |
| Memory usage | `memory_profiler` | `python -m memory_profiler script.py` |
| Production sampling | `py-spy` | `py-spy record -o flame.svg --pid PID` |
| Memory leaks | `tracemalloc` | Built-in, snapshot comparison |
| Benchmarking | `pytest-benchmark` | `pytest --benchmark-compare` |

## Profiling workflow
1. **Measure first** -- never optimize without a profile
2. `cProfile` to find hot functions (sort by `cumtime`)
3. `line_profiler` on the hot function to find hot lines
4. Fix algorithmic issues before micro-optimizations
5. Re-profile to verify improvement

## tracemalloc for leak detection
```python
tracemalloc.start()
snap1 = tracemalloc.take_snapshot()
# ... run suspect code ...
snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, 'lineno')[:10]:
    print(stat)
```

# Performance Patterns

## Caching decisions

| Scenario | Use |
|----------|-----|
| Pure function, small args | `@functools.lru_cache(maxsize=256)` |
| Pure function, unhashable args | `@functools.cache` (3.9+) or serialize key |
| TTL-based | `cachetools.TTLCache` or Redis |
| Async | `aiocache` or manual dict + asyncio.Lock |
| Cross-process | Redis or memcached |

**Gotcha**: `@lru_cache` requires hashable args -- use `tuple` not `dict`.

## __slots__ for many instances
```python
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y
# ~40% less memory per instance vs regular class
```

## Batch I/O operations
```python
# SLOW: commit per insert
for item in items:
    cursor.execute("INSERT ...", item)
    conn.commit()

# FAST: single commit
cursor.executemany("INSERT ...", items)
conn.commit()
```
