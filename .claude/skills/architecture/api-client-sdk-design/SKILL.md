---
name: api-client-sdk-design
description: "Use when building Python or TypeScript SDKs for APIs, implementing retry/backoff, pagination, auth flows, or generating clients from OpenAPI specs."
---

# API Client SDK Design

## SDK Architecture Decision Table

| Component | Purpose | Pattern |
|-----------|---------|---------|
| Client class | Entry point, holds config/auth | Singleton-ish, injectable |
| Resource classes | Group related endpoints | `client.users.list()` |
| Models | Request/response typing | Pydantic (Python), Zod (TS) |
| Transport | HTTP layer abstraction | Swappable (httpx, fetch) |
| Auth | Token management | Middleware/interceptor |
| Retry | Transient failure handling | Exponential backoff |
| Pagination | Iterator over paged results | Async iterator |

## SDK Architecture

### Python

```python
from __future__ import annotations
import httpx
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ClientConfig:
    base_url: str = "https://api.example.com/v1"
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3

class ExampleClient:
    """Top-level client. Resources are lazy-loaded attributes."""

    def __init__(self, config: ClientConfig | None = None, **kwargs):
        self._config = config or ClientConfig(**kwargs)
        self._http = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout,
            headers=self._default_headers(),
        )
        # Resource namespaces
        self.users = UsersResource(self)
        self.projects = ProjectsResource(self)

    def _default_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "example-sdk-python/0.1.0"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Central request method -- all retries, error handling here."""
        return _request_with_retry(self._http, method, path, self._config.max_retries, **kwargs)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class UsersResource:
    def __init__(self, client: ExampleClient):
        self._client = client

    def get(self, user_id: str) -> User:
        resp = self._client._request("GET", f"/users/{user_id}")
        return User(**resp.json())

    def list(self, **params) -> PageIterator[User]:
        return PageIterator(self._client, "/users", User, params)

    def create(self, *, name: str, email: str) -> User:
        resp = self._client._request("POST", "/users", json={"name": name, "email": email})
        return User(**resp.json())
```

### TypeScript

```typescript
interface ClientConfig {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
}

class ExampleClient {
  readonly users: UsersResource;
  readonly projects: ProjectsResource;
  private config: Required<ClientConfig>;

  constructor(config: ClientConfig = {}) {
    this.config = {
      baseUrl: config.baseUrl ?? "https://api.example.com/v1",
      apiKey: config.apiKey ?? "",
      timeout: config.timeout ?? 30_000,
      maxRetries: config.maxRetries ?? 3,
    };
    this.users = new UsersResource(this);
    this.projects = new ProjectsResource(this);
  }

  async _request<T>(method: string, path: string, opts?: RequestInit & { json?: unknown }): Promise<T> {
    const url = `${this.config.baseUrl}${path}`;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
    };
    const body = opts?.json ? JSON.stringify(opts.json) : undefined;

    return requestWithRetry<T>(url, { ...opts, method, headers, body }, this.config.maxRetries);
  }
}
```

## Auth Patterns

### API Key (simplest)

```python
# Set once in client config, sent on every request
headers["Authorization"] = f"Bearer {api_key}"
# or
headers["X-API-Key"] = api_key
```

### OAuth2 with Token Refresh

```python
import time
from dataclasses import dataclass

@dataclass
class TokenInfo:
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp

class OAuth2Auth:
    def __init__(self, client_id: str, client_secret: str, token_url: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._token: TokenInfo | None = None

    def get_token(self, http: httpx.Client) -> str:
        if self._token and self._token.expires_at > time.time() + 60:
            return self._token.access_token
        return self._refresh(http)

    def _refresh(self, http: httpx.Client) -> str:
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._token.refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        resp = http.post(self._token_url, data=payload)
        resp.raise_for_status()
        data = resp.json()
        self._token = TokenInfo(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", self._token.refresh_token),
            expires_at=time.time() + data["expires_in"],
        )
        return self._token.access_token
```

## Retry with Exponential Backoff

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
        try:
            resp = http.request(method, path, **kwargs)
            if resp.status_code not in RETRYABLE_STATUS:
                _raise_for_status(resp)
                return resp
            last_exc = APIError.from_response(resp)
        except httpx.TransportError as exc:
            last_exc = ConnectionError(str(exc))

        if attempt < max_retries:
            sleep = _backoff_delay(attempt, resp if 'resp' in dir() else None)
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

```typescript
// TypeScript equivalent
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

## Pagination Patterns

### Cursor-Based Async Iterator (Python)

```python
from typing import TypeVar, Generic, AsyncIterator
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class PageIterator(Generic[T]):
    """Sync iterator over paginated results."""

    def __init__(self, client, path: str, model: type[T], params: dict):
        self._client = client
        self._path = path
        self._model = model
        self._params = params
        self._cursor: str | None = None
        self._done = False
        self._buffer: list[T] = []

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if not self._buffer:
            if self._done:
                raise StopIteration
            self._fetch_page()
        if not self._buffer:
            raise StopIteration
        return self._buffer.pop(0)

    def _fetch_page(self):
        params = {**self._params}
        if self._cursor:
            params["cursor"] = self._cursor
        resp = self._client._request("GET", self._path, params=params)
        data = resp.json()
        self._buffer = [self._model(**item) for item in data["items"]]
        self._cursor = data.get("next_cursor")
        if not self._cursor:
            self._done = True
```

### TypeScript Async Iterator

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

## Error Handling

```python
class APIError(Exception):
    """Base SDK error."""
    def __init__(self, status: int, message: str, code: str | None = None):
        self.status = status
        self.message = message
        self.code = code
        super().__init__(f"[{status}] {code or 'unknown'}: {message}")

    @classmethod
    def from_response(cls, resp: httpx.Response) -> "APIError":
        try:
            body = resp.json()
            msg = body.get("message", resp.text)
            code = body.get("code")
        except Exception:
            msg = resp.text
            code = None

        status_map = {
            401: AuthenticationError,
            403: PermissionError_,
            404: NotFoundError,
            422: ValidationError_,
            429: RateLimitError,
        }
        klass = status_map.get(resp.status_code, cls)
        return klass(resp.status_code, msg, code)

class AuthenticationError(APIError): pass
class PermissionError_(APIError): pass
class NotFoundError(APIError): pass
class ValidationError_(APIError): pass
class RateLimitError(APIError):
    @property
    def retry_after(self) -> float | None:
        # Parsed from response headers during construction
        return getattr(self, "_retry_after", None)
```

## OpenAPI Codegen Tools

| Tool | Languages | Strengths | Weaknesses |
|------|-----------|-----------|------------|
| openapi-generator | 40+ | Broadest language support | Verbose output, heavy templates |
| Fern | Python, TS, Go, Java | Clean SDKs, good DX | SaaS pricing for advanced features |
| Speakeasy | Python, TS, Go | Polished output, retries built-in | Commercial |
| Stainless | Python, TS | Used by OpenAI/Anthropic | Limited access |
| oapi-codegen | Go only | Idiomatic Go | Go-only |

**Recommendation**: For internal APIs, hand-write the SDK. For public APIs with 20+ endpoints, use codegen. Stainless or Fern produce the cleanest output.

## Versioning Strategy

```
# URL versioning (most common for SDKs)
base_url = "https://api.example.com/v2"

# Header versioning (Stripe pattern)
headers["API-Version"] = "2024-01-15"

# SDK version != API version
# SDK version: semver (1.2.3)
# API version: date-based (2024-01-15) or integer (v2)
```

**SDK version bumps**:
- Patch: bug fixes, no API change
- Minor: new endpoints/fields (backward compatible)
- Major: breaking changes (removed fields, changed types)

## Testing SDKs

```python
# 1. Unit tests with respx (httpx mock)
import respx

@respx.mock
def test_get_user():
    respx.get("https://api.example.com/v1/users/123").mock(
        return_value=httpx.Response(200, json={"id": "123", "name": "Alice"})
    )
    client = ExampleClient(api_key="test")
    user = client.users.get("123")
    assert user.name == "Alice"

@respx.mock
def test_retry_on_503():
    route = respx.get("https://api.example.com/v1/users/123")
    route.side_effect = [
        httpx.Response(503),
        httpx.Response(200, json={"id": "123", "name": "Alice"}),
    ]
    client = ExampleClient(api_key="test")
    user = client.users.get("123")
    assert route.call_count == 2

# 2. Contract tests against recorded fixtures
# Record: responses saved as JSON fixtures
# Replay: mock HTTP with fixtures, assert SDK parses correctly

# 3. Integration test (run against staging)
def test_integration_create_and_delete():
    client = ExampleClient(api_key=os.environ["STAGING_API_KEY"])
    user = client.users.create(name="Test", email="test@example.com")
    assert user.id
    client.users.delete(user.id)
```

## Gotchas

- **Mutable default headers**: never do `headers={}` in function args; use `None` + create inside
- **Connection pooling missed**: create one `httpx.Client` per SDK instance, not per request; same for `fetch` with keep-alive
- **No timeout default**: always set a default timeout; 30s is reasonable; users can override
- **Retry on POST**: only retry idempotent requests by default; POST retries need idempotency keys
- **Token refresh race**: multiple threads refreshing simultaneously; use a lock or single-flight pattern
- **Pagination buffer memory**: don't load all pages into memory; use lazy iterators
- **Missing User-Agent**: always send SDK name + version; API providers use this for debugging and deprecation notices
- **Swallowing errors**: never catch and log silently; surface typed exceptions to SDK users
- **Forgetting `close()`**: HTTP clients hold connections; support context managers (`with` / `using`)
- **Versioning the SDK separately from the API**: they drift; document which API version each SDK version targets
