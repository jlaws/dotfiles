
# React State Management

## Selection Criteria

| Type | Solutions |
|------|-----------|
| **Local State** | useState, useReducer |
| **Global State** | Redux Toolkit, Zustand, Jotai |
| **Server State** | React Query, SWR, RTK Query |
| **URL State** | React Router, nuqs |
| **Form State** | React Hook Form, Formik |

```
Small app, simple state       -> Zustand or Jotai
Large app, complex state      -> Redux Toolkit
Heavy server interaction      -> React Query + light client state
Atomic/granular updates       -> Jotai
```

## Zustand

Create stores with `create<State>()`. Wrap with `devtools()` and `persist()` middleware. Use slice pattern with `StateCreator` for scalable stores. Select specific state to prevent re-renders: `useStore(s => s.user)`.

### Code Example: Zustand with Middleware Stack

```typescript
import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';

interface User {
  id: string;
  name: string;
  preferences: { theme: 'light' | 'dark'; notifications: boolean };
}

interface AppStore {
  user: User | null;
  setUser: (user: User) => void;
  updateTheme: (theme: 'light' | 'dark') => void;
  logout: () => void;
}

export const useAppStore = create<AppStore>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set) => ({
          user: null,
          setUser: (user) => set({ user }),
          updateTheme: (theme) =>
            set((state) => {
              if (state.user) {
                state.user.preferences.theme = theme;
              }
            }),
          logout: () => set({ user: null }),
        }))
      ),
      {
        name: 'app-store',
        partialize: (state) => ({ user: state.user }), // Only persist user
      }
    )
  )
);

// Usage: selects only user, prevents re-render on theme change
const user = useAppStore((s) => s.user);

// Subscribe outside React
useAppStore.subscribe(
  (state) => state.user,
  (user) => console.log('User changed:', user)
);

// Test: access state directly
const store = useAppStore.getState();
console.log(store.user);
```

## Redux Toolkit

Use `configureStore` + `createSlice`. Type `RootState` and `AppDispatch` from store. Use `createAsyncThunk` for async operations with `pending/fulfilled/rejected` matchers.

### Code Example: Redux Toolkit with Async Thunk

```typescript
import { configureStore, createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Async thunk for fetching user
export const fetchUser = createAsyncThunk(
  'user/fetchUser',
  async (userId: string) => {
    const res = await fetch(`/api/users/${userId}`);
    return res.json();
  }
);

const userSlice = createSlice({
  name: 'user',
  initialState: {
    data: null as any,
    loading: false,
    error: null as string | null,
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchUser.fulfilled, (state, action) => {
        state.loading = false;
        state.data = action.payload;
      })
      .addCase(fetchUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch';
      });
  },
});

export const store = configureStore({
  reducer: {
    user: userSlice.reducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Usage in component
import { useDispatch, useSelector } from 'react-redux';

export function UserDetail({ userId }: { userId: string }) {
  const dispatch = useDispatch<AppDispatch>();
  const { data, loading, error } = useSelector((state: RootState) => state.user);

  useEffect(() => {
    dispatch(fetchUser(userId));
  }, [userId]);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return <div>{data?.name}</div>;
}
```

## Jotai

Use `atom()` for base state, `atom(get => ...)` for derived, `atomWithStorage` for persistence. Write-only action atoms: `atom(null, (get, set) => ...)`.

### Code Example: Jotai Atoms and Composition

```typescript
import { atom } from 'jotai';
import { atomWithStorage } from 'jotai/storage';
import { useAtom, useAtomValue, useSetAtom } from 'jotai';

// Base atoms
const countAtom = atom(0);
const nameAtom = atom('');

// Persisted atom
const themeAtom = atomWithStorage<'light' | 'dark'>('theme', 'light');

// Derived atom (read-only)
const doubledCountAtom = atom((get) => {
  const count = get(countAtom);
  return count * 2;
});

// Write-only action atom
const incrementCountAtom = atom(
  null,
  (get, set) => {
    const current = get(countAtom);
    set(countAtom, current + 1);
  }
);

// Component: read doubledCountAtom, write via incrementCountAtom
export function Counter() {
  const doubled = useAtomValue(doubledCountAtom);
  const setIncrement = useSetAtom(incrementCountAtom);

  return (
    <div>
      <div>Doubled: {doubled}</div>
      <button onClick={() => setIncrement()}>Increment</button>
    </div>
  );
}

// Component: read and write theme
export function ThemeToggle() {
  const [theme, setTheme] = useAtom(themeAtom);
  return (
    <button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
      Theme: {theme}
    </button>
  );
}
```

## React Query

Use query key factories for cache organization. `useMutation` with `onMutate` for optimistic updates: cancel queries, snapshot previous, set optimistic data, rollback `onError`, invalidate `onSettled`.

### Code Example: React Query Mutation with Optimistic Update

```typescript
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

const todosQueryKey = ['todos'] as const;

export function TodoList() {
  const queryClient = useQueryClient();
  const { data: todos = [] } = useQuery({
    queryKey: todosQueryKey,
    queryFn: async () => (await fetch('/api/todos')).json(),
  });

  const addTodoMutation = useMutation({
    mutationFn: async (text: string) => {
      const res = await fetch('/api/todos', {
        method: 'POST',
        body: JSON.stringify({ text }),
        headers: { 'Content-Type': 'application/json' },
      });
      return res.json();
    },
    onMutate: async (newTodoText) => {
      // Cancel ongoing queries to avoid overwrite
      await queryClient.cancelQueries({ queryKey: todosQueryKey });

      // Snapshot previous state
      const previousTodos = queryClient.getQueryData<typeof todos>(todosQueryKey);

      // Set optimistic data
      if (previousTodos) {
        queryClient.setQueryData(todosQueryKey, [
          ...previousTodos,
          { id: 'temp', text: newTodoText, completed: false },
        ]);
      }

      return { previousTodos }; // Pass context to error handler
    },
    onError: (_, __, context) => {
      // Rollback on error
      if (context?.previousTodos) {
        queryClient.setQueryData(todosQueryKey, context.previousTodos);
      }
    },
    onSettled: () => {
      // Refetch to ensure sync with server
      queryClient.invalidateQueries({ queryKey: todosQueryKey });
    },
  });

  return (
    <div>
      {todos.map((todo) => (
        <div key={todo.id}>{todo.text}</div>
      ))}
      <button onClick={() => addTodoMutation.mutate('New todo')}>
        {addTodoMutation.isPending ? 'Adding...' : 'Add'}
      </button>
    </div>
  );
}
```

## Combining Client + Server State

Zustand for UI state (sidebar, modal), React Query for server state. Never duplicate server data in client stores.

## Anti-Patterns

- **Storing server data in Redux**: Server state = React Query/SWR; Redux = UI state only. Duplicating data creates sync bugs
- **Selector without memoization**: Always use `(state) => state.field` to extract; avoid object spread which recreates on every render
- **Putting complex business logic in slices**: Use thunks/services, keep reducers pure
- **useContext for global state at scale**: Context re-renders entire tree; use Zustand/Redux for large apps
- **Atom overuse in Jotai**: Derived atoms can trigger cascades; use `selectAtom` for fine-grained selection

## Cross-References

- **frontend:nextjs-app-router-patterns** -- server/client state boundary, React Server Components context
- **frontend:form-patterns** -- form state management, React Hook Form integration
- **frontend:i18n-and-localization** -- locale state management, context providers
