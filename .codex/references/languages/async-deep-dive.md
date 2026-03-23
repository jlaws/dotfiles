# Async Python Deep Dive

Extended patterns and code examples for asyncio. See parent SKILL.md for decision tables and core patterns.

## Async Context Managers

```python
from contextlib import asynccontextmanager
from typing import AsyncIterator

class DatabasePool:
    """Async resource that needs setup/teardown."""

    async def connect(self):
        self._pool = await create_pool(dsn="postgres://...")

    async def close(self):
        await self._pool.close()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.close()

# Or with decorator:
@asynccontextmanager
async def get_db_connection() -> AsyncIterator:
    pool = await create_pool(dsn="postgres://...")
    try:
        async with pool.acquire() as conn:
            yield conn
    finally:
        await pool.close()
```

## Async Generators and Iteration

```python
from typing import AsyncIterator

async def stream_records(query: str, batch_size: int = 100) -> AsyncIterator[dict]:
    """Async generator -- yields items lazily from paginated source."""
    offset = 0
    while True:
        rows = await db.fetch(query, limit=batch_size, offset=offset)
        if not rows:
            break
        for row in rows:
            yield dict(row)
        offset += batch_size

# Consuming:
async for record in stream_records("SELECT * FROM events"):
    process(record)

# Collecting with comprehension:
results = [r async for r in stream_records("SELECT * FROM events") if r["active"]]
```

### Async Iterator Protocol

```python
class AsyncChunkedReader:
    """Implements async iterator protocol directly."""

    def __init__(self, stream, chunk_size: int = 8192):
        self._stream = stream
        self._chunk_size = chunk_size

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        chunk = await self._stream.read(self._chunk_size)
        if not chunk:
            raise StopAsyncIteration
        return chunk

# Usage
async for chunk in AsyncChunkedReader(response.stream):
    await output.write(chunk)
```

## Async in ML Serving

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI

app = FastAPI()

# CPU-bound model inference should NOT run on the event loop
_executor = ProcessPoolExecutor(max_workers=4)

def _predict_sync(input_data: dict) -> dict:
    """Runs in a separate process -- no GIL contention."""
    model = get_cached_model()
    result = model.predict(input_data["features"])
    return {"prediction": result.tolist()}

@app.post("/predict")
async def predict(input_data: dict) -> dict:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(_executor, _predict_sync, input_data)
    return result
```

### Batched Async Inference Server

```python
import asyncio
from dataclasses import dataclass, field

@dataclass
class InferenceRequest:
    input: dict
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )

class AsyncBatchPredictor:
    """Collects requests, batches them, runs inference on batch."""

    def __init__(self, model, max_batch: int = 32, max_wait_ms: float = 5):
        self.model = model
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()

    async def start(self):
        """Start the background batch processing loop."""
        asyncio.create_task(self._batch_loop())

    async def predict(self, input_data: dict) -> dict:
        """Called per-request. Returns when batch containing this request completes."""
        req = InferenceRequest(input=input_data)
        await self._queue.put(req)
        return await req.future

    async def _batch_loop(self):
        while True:
            batch: list[InferenceRequest] = []

            # Wait for first request
            req = await self._queue.get()
            batch.append(req)

            # Collect more for up to max_wait
            deadline = asyncio.get_event_loop().time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(req)
                except TimeoutError:
                    break

            # Run batched inference
            try:
                inputs = [r.input for r in batch]
                results = self.model.predict_batch(inputs)
                for req, result in zip(batch, results):
                    req.future.set_result(result)
            except Exception as exc:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)
```

## TaskGroup Exception Handling

```python
async def fetch_with_partial_failure(urls: list[str]) -> tuple[list[dict], list[str]]:
    """Allow partial failures -- don't cancel everything."""
    results = []
    errors = []

    async with asyncio.TaskGroup() as tg:
        for url in urls:
            tg.create_task(_safe_fetch(url, results, errors))

    return results, errors

async def _safe_fetch(url: str, results: list, errors: list):
    """Catch inside the task so TaskGroup doesn't cancel siblings."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            results.append(resp.json())
    except Exception as exc:
        errors.append(f"{url}: {exc}")
```

## Testing Async Code

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_fetch():
    result = await fetch_data("https://api.example.com/data")
    assert result["status"] == "ok"

# Fixture that provides async resource
@pytest.fixture
async def db_conn():
    conn = await connect_db()
    yield conn
    await conn.close()

@pytest.mark.asyncio
async def test_query(db_conn):
    rows = await db_conn.fetch("SELECT 1")
    assert len(rows) == 1

# Testing timeouts
@pytest.mark.asyncio
async def test_timeout_behavior():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await asyncio.sleep(10)

# Testing concurrent behavior
@pytest.mark.asyncio
async def test_semaphore_limits_concurrency():
    active = 0
    max_active = 0

    async def track_concurrency(sem: asyncio.Semaphore):
        nonlocal active, max_active
        async with sem:
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1

    sem = asyncio.Semaphore(3)
    await asyncio.gather(*[track_concurrency(sem) for _ in range(10)])
    assert max_active <= 3
```

### pytest-asyncio Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # No need for @pytest.mark.asyncio on every test
```
