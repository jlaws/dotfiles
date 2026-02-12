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

**Pagination**: Use offset-based for REST (`page`, `page_size` params), cursor-based (Relay connections) for GraphQL. Return `total`, `pages`, `pageInfo` metadata.

**Error handling**: Standardize error responses: `{ error, message, details, timestamp, path }`. Use specific HTTP status codes. FastAPI: raise `HTTPException` with structured `detail`.

**HATEOAS**: Include `_links` dict with `self`, related resources, and available actions (`{ href, method }`).

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

**Resolver**: Decode cursor to offset, fetch `first + 1` to detect `hasNextPage`, encode offsets as cursors.

**DataLoader**: Create per-request DataLoader instances. `batch_load_fn` receives collected IDs, returns results in same order. Same pattern for 1:1 and 1:many relations.

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
