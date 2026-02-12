---
name: graphql-client-patterns
description: "Use when integrating GraphQL APIs in frontend apps. Covers client selection, code generation, cache normalization, optimistic updates, and subscriptions."
---

# GraphQL Client Patterns

## Client Selection

| Criteria | Apollo Client | urql | graphql-request | Relay |
|---|---|---|---|---|
| Cache | Normalized | Document/normalized | None | Normalized |
| Bundle size | ~35kb | ~8kb | ~3kb | ~30kb |
| SSR | Built-in | Built-in | Manual | Built-in |
| Subscriptions | Yes | Yes | No | Yes |
| Learning curve | Medium | Low | Minimal | High |
| Best for | Full-featured apps | Balanced needs | Simple fetching | Meta-scale apps |

**Default**: Apollo for complex apps with cache needs, urql for lighter footprint, graphql-request for simple query-only use cases.

## Code Generation

Use `@graphql-codegen/cli` with `client` preset. Define schema URL + document glob. Colocate `.graphql` files with features. Generated types flow into `useQuery(DocumentNode)`.

## Cache Normalization (Apollo)

Apollo `InMemoryCache` with `typePolicies`: define `keyArgs` for paginated fields, custom `merge` for appending, computed `read` fields.

## Optimistic Updates

Pass `optimisticResponse` to `useMutation`. In `update`, use `cache.modify` to insert the optimistic entry. Rollback `onError` with snapshotted previous data.

## Subscription Handling

Same `cache.modify` pattern as optimistic updates. Use `onData` callback to merge incoming subscription data into cache.

## Gotchas

- **N+1 on server**: Use DataLoader per-request; client cannot solve this
- **Cache invalidation**: `refetchQueries` is simpler than manual `cache.modify` -- use it unless perf-critical
- **Fragment colocation**: Keep fragments next to components that consume them; avoids stale field selections
- **Over-fetching**: Use `@defer` directive for heavy fields; split queries by viewport priority
- **SSR hydration mismatch**: Extract and rehydrate cache state; Apollo's `getDataFromTree` or urql's `ssrExchange`

## Cross-References

- **architecture:api-design-principles** -- GraphQL schema design and API conventions
- **architecture:api-client-sdk-design** -- Client abstraction layers and retry/error handling
