# Async Patterns

When to reach for async -- and when not to:

| Workload | Async? | Why |
|----------|--------|-----|
| HTTP API calls (many concurrent) | **Yes** | IO-bound, high concurrency wins |
| Database queries (connection pool) | **Yes** | IO-bound, pool management natural |
| File IO (many files) | **Maybe** | OS-level async varies; aiofiles helps |
| CPU-heavy computation | **No** | GIL blocks; use multiprocessing |
| ML model inference (GPU) | **Hybrid** | Offload to thread/process, await result |
| WebSocket server | **Yes** | Long-lived connections, perfect fit |
| CLI scripts | **Usually no** | Overhead not worth it for sequential tasks |

## gather vs TaskGroup
- `asyncio.gather(*tasks, return_exceptions=True)` -- fan-out, collect all results
- `asyncio.TaskGroup()` (3.11+) -- structured concurrency, cancels siblings on first exception
- **Prefer `TaskGroup`** for correctness; use `gather` when you need partial results

## Semaphore for rate limiting
```python
sem = asyncio.Semaphore(10)
async def bounded_fetch(client: httpx.AsyncClient, url: str) -> dict:
    async with sem:
        resp = await client.get(url)
        return resp.json()
```

## Timeouts (3.11+)
```python
async def fetch_with_timeout(url: str, timeout_s: float = 5.0) -> dict:
    try:
        async with asyncio.timeout(timeout_s):
            async with httpx.AsyncClient() as client:
                return (await client.get(url)).json()
    except TimeoutError:
        return {"error": f"Timeout after {timeout_s}s"}
```

## Cheat Sheet
```python
# Offload sync IO to thread
result = await asyncio.to_thread(sync_function, arg1, arg2)

# Offload CPU to process pool
result = await loop.run_in_executor(process_pool, cpu_function, arg)

# Fire and forget (use sparingly)
task = asyncio.create_task(background_work())
task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

# Wait for first completed
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
for t in pending: t.cancel()

# Async queue (producer/consumer)
queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
```

## Async Gotchas
- **Blocking the loop**: `time.sleep()`, `requests.get()`, or any sync IO in async freezes all tasks; use `asyncio.to_thread()` or async libs
- **GIL and CPU work**: async does NOT bypass GIL; use `ProcessPoolExecutor` for CPU
- **Forgetting `await`**: `result = async_func()` returns a coroutine, not the result
- **Exception swallowing in `gather`**: `return_exceptions=True` silently returns exceptions as values; check `isinstance(result, Exception)`
- **Async generators not closed**: break from `async for` early? use `async with aclosing(gen)` from `contextlib`
- **Event loop already running**: `asyncio.run()` inside running loop (e.g., Jupyter) fails; use `nest_asyncio` or `await` directly
- **Shared mutable state**: no GIL protection between `await` points; use `asyncio.Lock` if tasks mutate shared state
- **Cancellation**: always catch `CancelledError`, clean up, then re-raise
- **Mixing sync/async ORMs**: SQLAlchemy async requires `AsyncSession`; can't use sync session without `run_in_executor`

> **Deep dive**: async context managers, generators, ML serving, batched inference, testing -- see `references/async-deep-dive.md`
