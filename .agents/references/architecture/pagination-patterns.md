# Pagination Patterns

Cursor-based pagination implementations for API client SDKs using lazy iterators.

## Cursor-Based Async Iterator (Python)

```python
from typing import TypeVar, Generic, AsyncIterator
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class PageIterator(Generic[T]):
    """Async iterator over paginated results."""

    def __init__(self, client, path: str, model: type[T], params: dict):
        self._client = client
        self._path = path
        self._model = model
        self._params = params
        self._cursor: str | None = None
        self._done = False
        self._buffer: list[T] = []

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        if not self._buffer:
            if self._done:
                raise StopAsyncIteration
            await self._fetch_page()
        if not self._buffer:
            raise StopAsyncIteration
        return self._buffer.pop(0)

    async def _fetch_page(self):
        params = {**self._params}
        if self._cursor:
            params["cursor"] = self._cursor
        resp = await self._client._request("GET", self._path, params=params)
        data = resp.json()
        self._buffer = [self._model(**item) for item in data["items"]]
        self._cursor = data.get("next_cursor")
        if not self._cursor:
            self._done = True
```

## TypeScript Async Iterator

```typescript
async function* paginate<T>(
  client: ExampleClient,
  path: string,
  params: Record<string, string> = {},
): AsyncGenerator<T> {
  let cursor: string | undefined;

  do {
    const query = cursor ? { ...params, cursor } : params;
    const data = await client._request<{ items: T[]; next_cursor?: string }>("GET", path, { params: query });

    for (const item of data.items) {
      yield item;
    }
    cursor = data.next_cursor;
  } while (cursor);
}

// Usage
for await (const user of client.users.list({ role: "admin" })) {
  console.log(user.name);
}
```

## Pagination Strategy Comparison

| Strategy | Pros | Cons | Use When |
|----------|------|------|----------|
| Cursor-based | Stable under inserts/deletes, performant | Can't jump to page N | Default choice |
| Offset/limit | Simple, supports page jumping | Skips/duplicates on mutation | Static data, admin UIs |
| Keyset | DB-efficient, stable | Complex multi-column keys | Large datasets, sorted |

## Key Design Rules

- Always use **lazy iterators** -- never load all pages into memory
- Buffer one page at a time; fetch next page only when buffer exhausts
- Expose both `for item in results` (one-at-a-time) and `results.pages()` (page-at-a-time) APIs
- Pass page size as configurable param with sensible default (20-100)
