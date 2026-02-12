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

## Code Generation (graphql-codegen)

```typescript
// codegen.ts
import type { CodegenConfig } from '@graphql-codegen/cli'

const config: CodegenConfig = {
  schema: 'http://localhost:4000/graphql',
  documents: ['src/**/*.graphql'],
  generates: {
    './src/gql/': {
      preset: 'client',
      config: { scalars: { DateTime: 'string', JSON: 'Record<string, unknown>' } },
    },
  },
}
export default config
```

```graphql
# src/features/users/GetUser.graphql
query GetUser($id: ID!) {
  user(id: $id) { id name email ...UserAvatar }
}
fragment UserAvatar on User { avatarUrl displayName }
```

```typescript
// Typed query — generated types flow automatically
import { useQuery } from '@apollo/client'
import { GetUserDocument } from '@/gql/graphql'

function UserProfile({ id }: { id: string }) {
  const { data, loading, error } = useQuery(GetUserDocument, { variables: { id } })
  // data.user is fully typed
}
```

## Cache Normalization (Apollo)

```typescript
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // Merge paginated results
        feed: { keyArgs: ['type'], merge: (existing = [], incoming, { args }) =>
          args?.offset === 0 ? incoming : [...existing, ...incoming],
        },
      },
    },
    User: {
      fields: {
        fullName: { read: (_, { readField }) =>
          `${readField('firstName')} ${readField('lastName')}`,
        },
      },
    },
  },
})
```

## Optimistic Updates

```typescript
const [addComment] = useMutation(AddCommentDocument, {
  optimisticResponse: {
    addComment: {
      __typename: 'Comment',
      id: `temp-${Date.now()}`,
      body: commentText,
      author: currentUser,
      createdAt: new Date().toISOString(),
    },
  },
  update: (cache, { data }) => {
    cache.modify({
      id: cache.identify({ __typename: 'Post', id: postId }),
      fields: { comments: (refs) => [...refs, cache.writeFragment({
        fragment: CommentFieldsFragmentDoc,
        data: data!.addComment,
      })] },
    })
  },
})
```

## Subscription Handling

```typescript
const { data } = useSubscription(OnMessageDocument, {
  variables: { channelId },
  onData: ({ client, data }) => {
    client.cache.modify({
      id: client.cache.identify({ __typename: 'Channel', id: channelId }),
      fields: { messages: (refs) => [...refs, client.cache.writeFragment({
        fragment: MessageFragmentDoc, data: data.data!.messageAdded,
      })] },
    })
  },
})
```

## Gotchas

- **N+1 on server**: Use DataLoader per-request; client cannot solve this
- **Cache invalidation**: `refetchQueries` is simpler than manual `cache.modify` -- use it unless perf-critical
- **Fragment colocation**: Keep fragments next to components that consume them; avoids stale field selections
- **Over-fetching**: Use `@defer` directive for heavy fields; split queries by viewport priority
- **SSR hydration mismatch**: Extract and rehydrate cache state; Apollo's `getDataFromTree` or urql's `ssrExchange`

## Cross-References

- **architecture:api-design-principles** -- GraphQL schema design and API conventions
- **architecture:api-client-sdk-design** -- Client abstraction layers and retry/error handling
