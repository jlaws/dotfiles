# GraphQL Schema Design Patterns

## Schema Organization

### Modular Schema Structure
```graphql
# user.graphql
type User {
  id: ID!
  email: String!
  name: String!
  posts: [Post!]!
}

extend type Query {
  user(id: ID!): User
  users(first: Int, after: String): UserConnection!
}

extend type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}

# post.graphql
type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
}

extend type Query {
  post(id: ID!): Post
}
```

## Type Design Patterns

### 1. Interfaces for Polymorphism
```graphql
interface Node {
  id: ID!
  createdAt: DateTime!
}

type User implements Node {
  id: ID!
  createdAt: DateTime!
  email: String!
}

type Post implements Node {
  id: ID!
  createdAt: DateTime!
  title: String!
}

type Query {
  node(id: ID!): Node
}
```

### 2. Unions for Heterogeneous Results
```graphql
union SearchResult = User | Post | Comment

type Query {
  search(query: String!): [SearchResult!]!
}

# Query example
{
  search(query: "graphql") {
    ... on User {
      name
      email
    }
    ... on Post {
      title
      content
    }
    ... on Comment {
      text
      author { name }
    }
  }
}
```

### 3. Input Types
```graphql
input CreateUserInput {
  email: String!
  name: String!
  password: String!
  profileInput: ProfileInput
}

input ProfileInput {
  bio: String
  avatar: String
  website: String
}

input UpdateUserInput {
  id: ID!
  email: String
  name: String
  profileInput: ProfileInput
}
```

## Pagination Patterns

### Relay Cursor Pagination (Recommended)
```graphql
type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type Query {
  users(
    first: Int
    after: String
    last: Int
    before: String
  ): UserConnection!
}

# Usage
{
  users(first: 10, after: "cursor123") {
    edges {
      cursor
      node {
        id
        name
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

### Offset Pagination (Simpler)
```graphql
type UserList {
  items: [User!]!
  total: Int!
  page: Int!
  pageSize: Int!
}

type Query {
  users(page: Int = 1, pageSize: Int = 20): UserList!
}
```

## Mutation Design Patterns

### 1. Input/Payload Pattern
```graphql
input CreatePostInput {
  title: String!
  content: String!
  tags: [String!]
}

type CreatePostPayload {
  post: Post
  errors: [Error!]
  success: Boolean!
}

type Error {
  field: String
  message: String!
  code: String!
}

type Mutation {
  createPost(input: CreatePostInput!): CreatePostPayload!
}
```

### 2. Optimistic Response Support
```graphql
type UpdateUserPayload {
  user: User
  clientMutationId: String
  errors: [Error!]
}

input UpdateUserInput {
  id: ID!
  name: String
  clientMutationId: String
}

type Mutation {
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
}
```

### 3. Batch Mutations
```graphql
input BatchCreateUserInput {
  users: [CreateUserInput!]!
}

type BatchCreateUserPayload {
  results: [CreateUserResult!]!
  successCount: Int!
  errorCount: Int!
}

type CreateUserResult {
  user: User
  errors: [Error!]
  index: Int!
}

type Mutation {
  batchCreateUsers(input: BatchCreateUserInput!): BatchCreateUserPayload!
}
```

## Field Design

### Arguments and Filtering
```graphql
type Query {
  posts(
    # Pagination
    first: Int = 20
    after: String

    # Filtering
    status: PostStatus
    authorId: ID
    tag: String

    # Sorting
    orderBy: PostOrderBy = CREATED_AT
    orderDirection: OrderDirection = DESC

    # Searching
    search: String
  ): PostConnection!
}

enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

enum PostOrderBy {
  CREATED_AT
  UPDATED_AT
  TITLE
}

enum OrderDirection {
  ASC
  DESC
}
```

### Computed Fields
```graphql
type User {
  firstName: String!
  lastName: String!
  fullName: String!  # Computed in resolver

  posts: [Post!]!
  postCount: Int!    # Computed, doesn't load all posts
}

type Post {
  likeCount: Int!
  commentCount: Int!
  isLikedByViewer: Boolean!  # Context-dependent
}
```

## Subscriptions

```graphql
type Subscription {
  postAdded: Post!

  postUpdated(postId: ID!): Post!

  userStatusChanged(userId: ID!): UserStatus!
}

type UserStatus {
  userId: ID!
  online: Boolean!
  lastSeen: DateTime!
}

# Client usage
subscription {
  postAdded {
    id
    title
    author {
      name
    }
  }
}
```

## Custom Scalars

```graphql
scalar DateTime
scalar Email
scalar URL
scalar JSON
scalar Money

type User {
  email: Email!
  website: URL
  createdAt: DateTime!
  metadata: JSON
}

type Product {
  price: Money!
}
```

## Directives

### Built-in Directives
```graphql
type User {
  name: String!
  email: String! @deprecated(reason: "Use emails field instead")
  emails: [String!]!

  # Conditional inclusion
  privateData: PrivateData @include(if: $isOwner)
}

# Query
query GetUser($isOwner: Boolean!) {
  user(id: "123") {
    name
    privateData @include(if: $isOwner) {
      ssn
    }
  }
}
```

### Custom Directives
```graphql
directive @auth(requires: Role = USER) on FIELD_DEFINITION

enum Role {
  USER
  ADMIN
  MODERATOR
}

type Mutation {
  deleteUser(id: ID!): Boolean! @auth(requires: ADMIN)
  updateProfile(input: ProfileInput!): User! @auth
}
```

## Error Handling

### Union Error Pattern
```graphql
type User {
  id: ID!
  email: String!
}

type ValidationError {
  field: String!
  message: String!
}

type NotFoundError {
  message: String!
  resourceType: String!
  resourceId: ID!
}

type AuthorizationError {
  message: String!
}

union UserResult = User | ValidationError | NotFoundError | AuthorizationError

type Query {
  user(id: ID!): UserResult!
}

# Usage
{
  user(id: "123") {
    ... on User {
      id
      email
    }
    ... on NotFoundError {
      message
      resourceType
    }
    ... on AuthorizationError {
      message
    }
  }
}
```

### Errors in Payload
```graphql
type CreateUserPayload {
  user: User
  errors: [Error!]
  success: Boolean!
}

type Error {
  field: String
  message: String!
  code: ErrorCode!
}

enum ErrorCode {
  VALIDATION_ERROR
  UNAUTHORIZED
  NOT_FOUND
  INTERNAL_ERROR
}
```

## N+1 Query Problem Solutions

### DataLoader Pattern
```python
from aiodataloader import DataLoader

class PostLoader(DataLoader):
    async def batch_load_fn(self, post_ids):
        posts = await db.posts.find({"id": {"$in": post_ids}})
        post_map = {post["id"]: post for post in posts}
        return [post_map.get(pid) for pid in post_ids]

# Resolver
@user_type.field("posts")
async def resolve_posts(user, info):
    loader = info.context["loaders"]["post"]
    return await loader.load_many(user["post_ids"])
```

### Query Depth Limiting
```python
from graphql import GraphQLError, ValidationRule
from graphql.language import OperationDefinitionNode

def depth_limit_validator(max_depth: int) -> type[ValidationRule]:
    """Build a graphql-core validation rule that rejects over-deep queries."""

    def node_depth(node, depth: int = 0) -> int:
        # Inline selections only; fragment spreads need their definitions resolved
        selection_set = getattr(node, "selection_set", None)
        if selection_set is None:
            return depth
        return max((node_depth(child, depth + 1)
                    for child in selection_set.selections), default=depth)

    class DepthLimitRule(ValidationRule):
        def enter_operation_definition(self, node: OperationDefinitionNode, *_args):
            depth = node_depth(node)
            if depth > max_depth:
                self.report_error(GraphQLError(
                    f"Query depth {depth} exceeds maximum {max_depth}", node))

    return DepthLimitRule

# Usage:
# from graphql import parse, specified_rules, validate
# errors = validate(schema, parse(query),
#                   rules=[*specified_rules, depth_limit_validator(10)])
```

### Query Complexity Analysis
```python
from graphql import GraphQLError, ValidationRule
from graphql.language import FieldNode, IntValueNode, OperationDefinitionNode

LIST_SIZE_ARGS = ("first", "last", "limit")

def complexity_limit_validator(max_complexity: int) -> type[ValidationRule]:
    """Cost = each field counts 1, multiplied by the list size it requests."""

    def field_cost(node: FieldNode) -> int:
        multiplier = 1
        for arg in node.arguments:
            if arg.name.value in LIST_SIZE_ARGS and isinstance(arg.value, IntValueNode):
                multiplier = int(arg.value.value)
        children = node.selection_set.selections if node.selection_set else []
        return multiplier * (1 + sum(field_cost(c) for c in children
                                     if isinstance(c, FieldNode)))

    class ComplexityLimitRule(ValidationRule):
        def enter_operation_definition(self, node: OperationDefinitionNode, *_args):
            total = sum(field_cost(f) for f in node.selection_set.selections
                        if isinstance(f, FieldNode))
            if total > max_complexity:
                self.report_error(GraphQLError(
                    f"Query complexity {total} exceeds maximum {max_complexity}", node))

    return ComplexityLimitRule
```

## Schema Versioning

### Field Deprecation
```graphql
type User {
  name: String! @deprecated(reason: "Use firstName and lastName")
  firstName: String!
  lastName: String!
}
```

### Schema Evolution
```graphql
# v1 - Initial
type User {
  name: String!
}

# v2 - Add optional field (backward compatible)
type User {
  name: String!
  email: String
}

# v3 - Deprecate and add new field
type User {
  name: String! @deprecated(reason: "Use firstName/lastName")
  firstName: String!
  lastName: String!
  email: String
}
```

## Best Practices Summary

1. **Nullable vs Non-Null**: Start nullable, make non-null when guaranteed
2. **Input Types**: Always use input types for mutations
3. **Payload Pattern**: Return errors in mutation payloads
4. **Pagination**: Use cursor-based for infinite scroll, offset for simple cases
5. **Naming**: Use camelCase for fields, PascalCase for types
6. **Deprecation**: Use `@deprecated` instead of removing fields
7. **DataLoaders**: Always use for relationships to prevent N+1
8. **Complexity Limits**: Protect against expensive queries
9. **Custom Scalars**: Use for domain-specific types (Email, DateTime)
10. **Documentation**: Document all fields with descriptions
