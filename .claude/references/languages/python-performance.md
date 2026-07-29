# Python Performance Patterns

Profiling, concurrency, vectorization, memory efficiency, and data structures.

## Profiling Strategy

| Need | Tool | Command |
|------|------|---------|
| Where is time spent? | `cProfile` | `python -m cProfile -o out.prof script.py` |
| Line-by-line timing | `line_profiler` | `kernprof -l -v script.py` |
| Memory usage | `memory_profiler` | `python -m memory_profiler script.py` |
| Production sampling | `py-spy` | `py-spy record -o flame.svg --pid PID` |
| Memory leaks | `tracemalloc` | Built-in, snapshot comparison |
| Benchmarking | `pytest-benchmark` | `pytest --benchmark-compare` |

### Profiling workflow
1. **Measure first** -- never optimize without a profile
2. `cProfile` to find hot functions (sort by `cumtime`)
3. `line_profiler` on the hot function to find hot lines
4. Fix algorithmic issues before micro-optimizations
5. Re-profile to verify improvement

### tracemalloc for leak detection
```python
tracemalloc.start()
snap1 = tracemalloc.take_snapshot()
# ... run suspect code ...
snap2 = tracemalloc.take_snapshot()
for stat in snap2.compare_to(snap1, 'lineno')[:10]:
    print(stat)
```

## GIL and Concurrency

| Scenario | Use | Why |
|----------|-----|-----|
| I/O-bound (network, disk) | `asyncio` or `threading` | GIL released during I/O |
| CPU-bound (math, data processing) | `multiprocessing` | Bypasses GIL, separate processes |
| Mixed workload | `asyncio` + `multiprocessing.Pool` | Async for I/O, pool for CPU tasks |
| Async + CPU work | `loop.run_in_executor()` | Offload CPU to thread pool |

```python
# SLOW: GIL blocks both threads on CPU work
import threading
def cpu_work():
    total = sum(i*i for i in range(10**8))
t1 = threading.Thread(target=cpu_work)
t2 = threading.Thread(target=cpu_work)
t1.start(); t2.start()
t1.join(); t2.join()  # ~2x slower than sequential

# FAST: Multiprocessing avoids GIL
from multiprocessing import Pool
with Pool(2) as p:
    p.map(cpu_work, [None, None])  # ~2x faster
```

**Gotcha**: `multiprocessing` has startup overhead; only worth it for >100ms CPU work.

## NumPy Vectorization

```python
# SLOW: Python loop
result = []
for i in range(len(data)):
    result.append(data[i] * 2 + 1)

# FAST: NumPy vectorization (100x faster)
import numpy as np
data = np.array(data)
result = data * 2 + 1  # operates on entire array at C speed
```

Vectorization rules:
- Avoid explicit Python loops; use `.apply()`, list comprehensions as fallback
- Keep operations in NumPy arrays (don't extract scalars mid-calculation)
- Use boolean indexing instead of filtering: `arr[arr > 0]`

## String Performance

```python
# SLOW: += creates new string each iteration
s = ""
for word in words:
    s += word  # O(n^2)

# FAST: join
s = "".join(words)  # O(n)

# f-strings vs format()
s = f"x={x}, y={y}"  # Python 3.6+, fastest, readable
s = "{} {}".format(x, y)  # 10-15% slower
s = "%s %s" % (x, y)  # 20% slower, deprecated
```

## Generators & Iterators (Memory Efficiency)

```python
# SLOW: list in memory
def read_file_slow(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(process(line))
    return lines  # O(n) memory

# FAST: generator (lazy evaluation)
def read_file_fast(path):
    with open(path) as f:
        for line in f:
            yield process(line)  # O(1) memory, computed on-demand

for result in read_file_fast("large.txt"):
    print(result)
```

- Use `yield` for large datasets or streaming.

## Data Structure Selection

| Structure | Lookup | Insert | Use Case |
|-----------|--------|--------|----------|
| `list` | O(n) | O(1) amortized | Ordered, few lookups |
| `dict` | O(1) | O(1) | Key-value, frequent lookups |
| `set` | O(1) | O(1) | Uniqueness, membership tests |
| `collections.defaultdict` | O(1) | O(1) | Nested dicts, grouped data |
| `collections.Counter` | O(1) | O(1) | Frequency counting |
| `sortedcontainers.SortedDict` | O(log n) | O(n) | Ordered + range queries |

```python
# SLOW: check membership in list
if item in large_list:  # O(n)
    ...

# FAST: use set
if item in large_set:  # O(1)
    ...

# defaultdict vs dict
from collections import defaultdict
freq = defaultdict(int)
freq[word] += 1  # no KeyError

# Counter for histograms
from collections import Counter
count = Counter(words)  # freq[word] = count
most_common = count.most_common(10)
```

## Caching Decisions

| Scenario | Use |
|----------|-----|
| Pure function, small args | `@functools.lru_cache(maxsize=256)` |
| Pure function, unhashable args | `@functools.cache` (3.9+) or serialize key |
| TTL-based | `cachetools.TTLCache` or Redis |
| Async | `aiocache` or manual dict + asyncio.Lock |
| Cross-process | Redis or memcached |

**Gotcha**: `@lru_cache` requires hashable args -- use `tuple` not `dict`.

## __slots__ for Many Instances

```python
class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y
# ~40% less memory per instance vs regular class
```

## Batch I/O Operations

```python
# SLOW: commit per insert
for item in items:
    cursor.execute("INSERT ...", item)
    conn.commit()

# FAST: single commit
cursor.executemany("INSERT ...", items)
conn.commit()
```
