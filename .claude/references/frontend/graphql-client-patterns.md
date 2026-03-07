
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

### Codegen Config Example

**codegen.ts:**
```typescript
import { CodegenConfig } from '@graphql-codegen/cli';

const config: CodegenConfig = {
  schema: 'https://api.example.com/graphql',
  documents: ['src/**/*.graphql'],
  generates: {
    'src/gql/': {
      preset: 'client',
      config: {
        useTypeImports: true,
        enumsAsTypes: true,
      },
    },
  },
  ignoreNoDocuments: true,
};

export default config;
```

## Apollo Client Setup

### Code Example: Apollo useQuery + useMutation with TypeScript

```typescript
import { gql, useQuery, useMutation, ApolloClient, InMemoryCache, HttpLink } from '@apollo/client';

// Query definition (src/queries/GetUser.graphql)
const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
    }
  }
`;

// Mutation definition
const UPDATE_USER = gql`
  mutation UpdateUser($id: ID!, $input: UserInput!) {
    updateUser(id: $id, input: $input) {
      id
      name
      email
    }
  }
`;

// Generated types from codegen
type GetUserQuery = {
  user: { id: string; name: string; email: string };
};

type UpdateUserMutation = {
  updateUser: { id: string; name: string; email: string };
};

// Component with useQuery
export function UserProfile({ userId }: { userId: string }) {
  const { data, loading, error } = useQuery<GetUserQuery>(GET_USER, {
    variables: { id: userId },
  });

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return <div>{data?.user.name}</div>;
}

// Component with useMutation
export function UserEditor({ userId }: { userId: string }) {
  const [updateUser, { loading }] = useMutation<UpdateUserMutation>(UPDATE_USER, {
    onCompleted: (data) => {
      console.log('Updated:', data.updateUser);
    },
    onError: (error) => {
      console.error('Failed:', error.message);
    },
  });

  const handleUpdate = () => {
    updateUser({
      variables: {
        id: userId,
        input: { name: 'New Name', email: 'new@example.com' },
      },
    });
  };

  return <button onClick={handleUpdate} disabled={loading}>Update</button>;
}
```

## Cache Normalization (Apollo)

Apollo `InMemoryCache` with `typePolicies`: define `keyArgs` for paginated fields, custom `merge` for appending, computed `read` fields.

### Code Example: Cache typePolicies for Pagination

```typescript
import { InMemoryCache, NormalizedCacheObject } from '@apollo/client';

const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // Pagination: concatenate pages, keyed by filter+sort
        posts: {
          keyArgs: ['filter', 'sort'],
          merge: (existing = [], incoming, { args }) => {
            // existing: previously cached pages
            // incoming: new page of results
            if (args?.offset === 0) {
              return incoming; // Reset if offset is 0 (refresh)
            }
            return [...existing, ...incoming];
          },
        },
      },
    },
    Post: {
      // Custom key: by default uses __typename + id
      keyFields: ['id'],
      fields: {
        // Computed field: derive author display from nested object
        authorDisplay: {
          read(_, { readField }) {
            const name = readField('author');
            return `By ${name}`;
          },
        },
      },
    },
  },
});
```

## Optimistic Updates

Pass `optimisticResponse` to `useMutation`. In `update`, use `cache.modify` to insert the optimistic entry. Rollback `onError` with snapshotted previous data.

### Code Example: Optimistic Update with Rollback

```typescript
import { useMutation, gql, ApolloCache, NormalizedCacheObject } from '@apollo/client';

const LIKE_POST = gql`
  mutation LikePost($postId: ID!) {
    likePost(postId: $postId) {
      id
      likes
      userHasLiked
    }
  }
`;

export function PostCard({ postId, initialLikes }: { postId: string; initialLikes: number }) {
  const [like, { loading }] = useMutation(LIKE_POST, {
    optimisticResponse: {
      likePost: {
        __typename: 'Post',
        id: postId,
        likes: initialLikes + 1,
        userHasLiked: true,
      },
    },
    update: (cache, { data }) => {
      // Manually update cache with mutation result
      cache.modify({
        fields: {
          posts(existing = []) {
            return existing.map(post =>
              post.id === postId ? data.likePost : post
            );
          },
        },
      });
    },
    onError: (error, variables, context) => {
      // Rollback: Apollo auto-reverts optimistic response on error
      console.error('Like failed:', error);
    },
  });

  return (
    <button onClick={() => like({ variables: { postId } })} disabled={loading}>
      Like ({initialLikes})
    </button>
  );
}
```

## Subscription Handling

Same `cache.modify` pattern as optimistic updates. Use `onData` callback to merge incoming subscription data into cache.

### Code Example: Subscription with Cache Update

```typescript
import { useSubscription, gql } from '@apollo/client';

const MESSAGE_ADDED = gql`
  subscription OnMessageAdded($roomId: ID!) {
    messageAdded(roomId: $roomId) {
      id
      text
      author
      createdAt
    }
  }
`;

export function ChatRoom({ roomId }: { roomId: string }) {
  const { data, loading } = useSubscription(MESSAGE_ADDED, {
    variables: { roomId },
    onData: ({ data }) => {
      // Merge new message into cache
      if (data.data?.messageAdded) {
        // Apollo automatically updates if cache is properly normalized
        // Otherwise, use cache.modify() to manually append
      }
    },
  });

  return (
    <div>
      {loading && 'Connecting...'}
      {/* Render messages from useQuery(GET_MESSAGES) */}
    </div>
  );
}
```

## urql Setup (Lightweight Alternative)

```typescript
import { createClient } from 'urql';

const client = createClient({
  url: 'https://api.example.com/graphql',
  exchanges: [
    // Default: cacheExchange (document cache), fetchExchange
  ],
});
```

## Gotchas

- **N+1 on server**: Use DataLoader per-request; client cannot solve this
- **Cache invalidation**: `refetchQueries` is simpler than manual `cache.modify` -- use it unless perf-critical
- **Fragment colocation**: Keep fragments next to components that consume them; avoids stale field selections
- **Over-fetching**: Use `@defer` directive for heavy fields; split queries by viewport priority
- **SSR hydration mismatch**: Extract and rehydrate cache state; Apollo's `getDataFromTree` or urql's `ssrExchange`

## Cross-References

- **architecture:api-design-principles** -- API design, SDK architecture, retry/error handling
