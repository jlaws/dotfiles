# Retry with Exponential Backoff

Retry logic for transient failures in API client SDKs. Both Python and TypeScript implementations.

## Retryable Status Codes

| Code | Meaning | Retry? |
|------|---------|--------|
| 408 | Request Timeout | Yes |
| 429 | Rate Limited | Yes (respect Retry-After) |
| 500 | Internal Server Error | Yes |
| 502 | Bad Gateway | Yes |
| 503 | Service Unavailable | Yes |
| 504 | Gateway Timeout | Yes |

**Important**: Only retry idempotent requests (GET, PUT, DELETE) by default. POST retries need idempotency keys.

## Python Implementation

```python
import time
import random
import httpx

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

def _request_with_retry(
    http: httpx.Client,
    method: str,
    path: str,
    max_retries: int,
    **kwargs,
) -> httpx.Response:
    last_exc = None
    for attempt in range(max_retries + 1):
        resp = None       # Reset each attempt: never read Retry-After off a stale response
        try:
            resp = http.request(method, path, **kwargs)
            if resp.status_code not in RETRYABLE_STATUS:
                _raise_for_status(resp)
                return resp
            last_exc = APIError.from_response(resp)
        except httpx.TransportError as exc:
            last_exc = ConnectionError(str(exc))

        if attempt < max_retries:
            sleep = _backoff_delay(attempt, resp)
            time.sleep(sleep)

    raise last_exc

def _backoff_delay(attempt: int, response: httpx.Response | None = None) -> float:
    """Exponential backoff with jitter. Respects Retry-After header."""
    if response and "Retry-After" in response.headers:
        return float(response.headers["Retry-After"])
    base = min(2 ** attempt, 30)  # Cap at 30 seconds
    jitter = random.uniform(0, base * 0.5)
    return base + jitter
```

## TypeScript Implementation

```typescript
const RETRYABLE_STATUS = new Set([408, 429, 500, 502, 503, 504]);

async function requestWithRetry<T>(
  url: string,
  init: RequestInit,
  maxRetries: number,
): Promise<T> {
  let lastError: Error | undefined;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const resp = await fetch(url, init);
      if (!RETRYABLE_STATUS.has(resp.status)) {
        if (!resp.ok) throw new APIError(resp.status, await resp.text());
        return (await resp.json()) as T;
      }
      lastError = new APIError(resp.status, await resp.text());

      const retryAfter = resp.headers.get("Retry-After");
      const delay = retryAfter ? parseFloat(retryAfter) * 1000 : backoffDelay(attempt);
      await sleep(delay);
    } catch (e) {
      if (e instanceof APIError) { lastError = e; continue; }
      throw e;
    }
  }
  throw lastError!;
}

function backoffDelay(attempt: number): number {
  const base = Math.min(2 ** attempt * 1000, 30_000);
  return base + Math.random() * base * 0.5;
}
```

## Backoff Strategy Summary

| Attempt | Base Delay | With Jitter (max) |
|---------|------------|-------------------|
| 0 | 1s | 1.5s |
| 1 | 2s | 3s |
| 2 | 4s | 6s |
| 3 | 8s | 12s |
| 4 | 16s | 24s |
| 5+ | 30s (cap) | 45s |
