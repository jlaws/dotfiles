---
name: api-design-principles
description: Master REST and GraphQL API design principles to build intuitive, scalable, and maintainable APIs that delight developers. Use when designing new APIs, reviewing API specifications, or establishing API design standards.
---

# API Design Principles

## REST API Design Patterns

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

### Pagination and Filtering

```python
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query
from typing import List, Optional

class PaginatedResponse(BaseModel):
    items: List[dict]
    total: int
    page: int
    page_size: int
    pages: int

@app.get("/api/users", response_model=PaginatedResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    query = build_query(status=status, search=search)
    total = await count_users(query)
    offset = (page - 1) * page_size
    users = await fetch_users(query, limit=page_size, offset=offset)
    return PaginatedResponse(
        items=users, total=total, page=page, page_size=page_size,
        pages=(total + page_size - 1) // page_size
    )
```

### Error Handling

```python
from fastapi import HTTPException, status
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[dict] = None
    timestamp: str
    path: str

def raise_not_found(resource: str, id: str):
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error": "NotFound", "message": f"{resource} not found", "details": {"id": id}}
    )

def raise_validation_error(errors):
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": "ValidationError", "message": "Request validation failed",
                "details": {"errors": [e.dict() for e in errors]}}
    )
```

### HATEOAS

```python
class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    _links: dict

    @classmethod
    def from_user(cls, user, base_url: str):
        return cls(
            id=user.id, name=user.name, email=user.email,
            _links={
                "self": {"href": f"{base_url}/api/users/{user.id}"},
                "orders": {"href": f"{base_url}/api/users/{user.id}/orders"},
                "update": {"href": f"{base_url}/api/users/{user.id}", "method": "PATCH"},
                "delete": {"href": f"{base_url}/api/users/{user.id}", "method": "DELETE"}
            }
        )
```

## GraphQL Design Patterns

### Schema Design

```graphql
type User {
  id: ID!
  email: String!
  name: String!
  createdAt: DateTime!
  orders(first: Int = 20, after: String, status: OrderStatus): OrderConnection!
  profile: UserProfile
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

type Query {
  user(id: ID!): User
  users(first: Int = 20, after: String, search: String): UserConnection!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
}
```

### Resolver with Cursor Pagination

```python
@query.field("users")
async def resolve_users(obj, info, first=20, after=None, search=None):
    offset = decode_cursor(after) if after else 0
    users = await fetch_users(limit=first + 1, offset=offset, search=search)
    has_next = len(users) > first
    if has_next:
        users = users[:first]
    edges = [{"node": user, "cursor": encode_cursor(offset + i)}
             for i, user in enumerate(users)]
    return {
        "edges": edges,
        "pageInfo": {
            "hasNextPage": has_next, "hasPreviousPage": offset > 0,
            "startCursor": edges[0]["cursor"] if edges else None,
            "endCursor": edges[-1]["cursor"] if edges else None
        },
        "totalCount": await count_users(search=search)
    }
```

### DataLoader (N+1 Prevention)

```python
from aiodataloader import DataLoader

class UserLoader(DataLoader):
    async def batch_load_fn(self, user_ids):
        users = await fetch_users_by_ids(user_ids)
        user_map = {user["id"]: user for user in users}
        return [user_map.get(uid) for uid in user_ids]

class OrdersByUserLoader(DataLoader):
    async def batch_load_fn(self, user_ids):
        orders = await fetch_orders_by_user_ids(user_ids)
        orders_by_user = {}
        for order in orders:
            orders_by_user.setdefault(order["user_id"], []).append(order)
        return [orders_by_user.get(uid, []) for uid in user_ids]

def create_context():
    return {"loaders": {"user": UserLoader(), "orders_by_user": OrdersByUserLoader()}}
```

### Persisted Queries

```graphql
# Client sends hash instead of full query
GET /graphql?extensions={"persistedQuery":{"version":1,"sha256Hash":"abc123..."}}
```

Benefits: smaller payloads, allowlisted queries only, CDN caching.

### Schema Federation

Apollo Federation composes subgraphs into a supergraph for microservices. Each service owns its types and extends shared entities via `@key` directives. The gateway/router handles query planning across services.

### GraphQL Error Conventions

- Use union return types for business errors:
  ```graphql
  union CreateUserResult = User | ValidationError | NotFoundError
  type ValidationError { field: String!; message: String! }
  ```
- Never throw from resolvers for business logic errors -- reserve exceptions for unexpected failures
- Return error types as part of the schema so clients get type-safe error handling

## API Versioning

```
URL:     /api/v1/users              (recommended - clear, easy to route)
Header:  Accept: application/vnd.api+json; version=1
Query:   /api/users?version=1
```

## Pitfalls

- **Over-fetching/Under-fetching (REST)**: Fixed in GraphQL but requires DataLoaders
- **Inconsistent Error Formats**: Standardize error responses
- **Ignoring HTTP Semantics**: POST for idempotent operations breaks expectations
- **Tight Coupling**: API structure shouldn't mirror database schema

## Cross-References

- **frontend:graphql-client-patterns** -- client-side GraphQL libraries, cache normalization, optimistic updates
- **architecture:api-client-sdk-design** -- SDK generation, client library patterns
