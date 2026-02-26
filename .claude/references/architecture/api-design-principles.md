# API Design Principles & Client SDK Patterns

## REST API Design

### Resource Collection Design

```python
# Resource-oriented endpoints
GET    /api/users              # List users (with pagination)
POST   /api/users              # Create user
GET    /api/users/{id}         # Get specific user
PUT    /api/users/{id}         # Replace user
PATCH  /api/users/{id}         # Update user fields
DELETE /api/users/{id}         # Delete user

# Nested resources
GET    /api/users/{id}/orders  # Get user's orders
POST   /api/users/{id}/orders  # Create order for user
```

**Pagination**: Offset-based for REST (`page`, `page_size`), cursor-based (Relay connections) for GraphQL. Return `total`, `pages`, `pageInfo` metadata.

**Error handling**: Standardize: `{ error, message, details, timestamp, path }`. Use specific HTTP status codes. FastAPI: raise `HTTPException` with structured `detail`.

**HATEOAS**: Include `_links` dict with `self`, related resources, and available actions (`{ href, method }`).

### API Versioning

```
URL:     /api/v1/users              (recommended - clear, easy to route)
Header:  Accept: application/vnd.api+json; version=1
Query:   /api/users?version=1
```

## GraphQL Design

### Schema Design

```graphql
type User {
  id: ID!
  email: String!
  name: String!
  createdAt: DateTime!
  orders(first: Int = 20, after: String, status: OrderStatus): OrderConnection!
}

# Relay-style pagination
type OrderConnection {
  edges: [OrderEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# Input/Payload mutation pattern
input CreateUserInput { email: String!; name: String!; password: String! }
type CreateUserPayload { user: User; errors: [Error!] }
type Error { field: String; message: String! }
```

**Resolver**: Decode cursor to offset, fetch `first + 1` to detect `hasNextPage`, encode offsets as cursors.

**DataLoader**: Per-request instances. `batch_load_fn` receives collected IDs, returns results in same order. Same pattern for 1:1 and 1:many.

**Persisted queries**: Client sends hash instead of full query. Benefits: smaller payloads, allowlisted queries, CDN caching.

**Federation**: Apollo Federation composes subgraphs via `@key` directives. Gateway handles query planning across services.

**Error conventions**: Use union return types for business errors (`union CreateUserResult = User | ValidationError | NotFoundError`). Never throw from resolvers for business logic -- reserve exceptions for unexpected failures.

## SDK Architecture

### Decision Table

| Component | Purpose | Pattern |
|-----------|---------|---------|
| Client class | Entry point, config/auth | Singleton-ish, injectable |
| Resource classes | Group related endpoints | `client.users.list()` |
| Models | Request/response typing | Pydantic (Python), Zod (TS) |
| Transport | HTTP layer abstraction | Swappable (httpx, fetch) |
| Auth | Token management | Middleware/interceptor |
| Retry | Transient failure handling | Exponential backoff |
| Pagination | Iterator over paged results | Async iterator |

### Python SDK

```python
from __future__ import annotations
import httpx
from dataclasses import dataclass

@dataclass
class ClientConfig:
    base_url: str = "https://api.example.com/v1"
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3

class ExampleClient:
    def __init__(self, config: ClientConfig | None = None, **kwargs):
        self._config = config or ClientConfig(**kwargs)
        self._http = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout,
            headers=self._default_headers(),
        )
        self.users = UsersResource(self)
        self.projects = ProjectsResource(self)

    def _default_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "example-sdk-python/0.1.0"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
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

### TypeScript SDK

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

### API Key

```python
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
    expires_at: float

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
        resp = http.post(self._token_url, data={
            "grant_type": "refresh_token",
            "refresh_token": self._token.refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        })
        resp.raise_for_status()
        data = resp.json()
        self._token = TokenInfo(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", self._token.refresh_token),
            expires_at=time.time() + data["expires_in"],
        )
        return self._token.access_token
```

## OpenAPI Codegen Tools

| Tool | Languages | Strengths |
|------|-----------|-----------|
| openapi-generator | 40+ | Broadest language support |
| Fern | Python, TS, Go, Java | Clean SDKs, good DX |
| Speakeasy | Python, TS, Go | Polished output, retries built-in |
| Stainless | Python, TS | Used by OpenAI/Anthropic |
| oapi-codegen | Go only | Idiomatic Go |

Hand-write SDKs for internal APIs. Use codegen (Stainless or Fern) for public APIs with 20+ endpoints.

## SDK Versioning

```
# URL versioning (most common)
base_url = "https://api.example.com/v2"

# Header versioning (Stripe pattern)
headers["API-Version"] = "2024-01-15"
```

SDK semver: patch = bug fix, minor = new endpoints (backward compatible), major = breaking changes.

## Pitfalls

**API design**: Over-fetching/under-fetching (GraphQL + DataLoaders fix this) -- Inconsistent error formats -- POST for idempotent operations -- API structure mirroring DB schema

**SDK implementation**: Mutable default headers (use `None` + create inside) -- Missing connection pooling (one `httpx.Client` per instance) -- No timeout default (30s reasonable) -- Retry on POST without idempotency keys -- Token refresh race condition (use a lock) -- Pagination loading all pages (use lazy iterators) -- Missing User-Agent header -- Swallowing errors -- Forgetting `close()` / context managers -- SDK vs API version drift

## References

Detailed implementations:

- [Retry patterns and exponential backoff](retry-patterns.md)
- [Pagination patterns](pagination-patterns.md)
- [SDK error handling catalog](error-handling-patterns.md)
- [SDK testing strategies](testing-strategies.md)
- [GraphQL schema design](graphql-schema-design.md)
- [REST best practices](rest-best-practices.md)

## Cross-References

- **frontend:graphql-client-patterns** -- client-side GraphQL libraries, cache normalization, optimistic updates
- **documentation:openapi-spec-generation** -- OpenAPI specs for SDK generation
