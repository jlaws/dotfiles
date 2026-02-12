---
name: react-state-management
description: Master modern React state management with Redux Toolkit, Zustand, Jotai, and React Query. Use when setting up global state, managing server state, or choosing between state management solutions.
---

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

## Zustand (Simplest)

```typescript
import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'

export const useStore = create<AppState>()(
  devtools(persist((set) => ({
    user: null,
    theme: 'light',
    setUser: (user) => set({ user }),
    toggleTheme: () => set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' })),
  }), { name: 'app-storage' }))
)
```

### Zustand with Slices (Scalable)

```typescript
export const createUserSlice: StateCreator<UserSlice & CartSlice, [], [], UserSlice> = (set, get) => ({
  user: null,
  isAuthenticated: false,
  login: async (credentials) => {
    const user = await authApi.login(credentials)
    set({ user, isAuthenticated: true })
  },
  logout: () => set({ user: null, isAuthenticated: false }),
})

// Combine slices
type StoreState = UserSlice & CartSlice
export const useStore = create<StoreState>()((...args) => ({
  ...createUserSlice(...args),
  ...createCartSlice(...args),
}))

// Selective subscriptions (prevents unnecessary re-renders)
export const useUser = () => useStore((state) => state.user)
```

### Zustand Advanced Patterns

```typescript
import { create } from 'zustand'
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

const useStore = create<AppState>()(
  devtools(
    subscribeWithSelector(
      persist(
        immer((set) => ({
          nested: { deep: { value: 0 } },
          updateDeep: () => set((state) => { state.nested.deep.value += 1 }),
        })),
        {
          name: 'app-storage',
          partialize: (state) => ({ nested: state.nested }), // persist subset
          merge: (persisted, current) => deepMerge(current, persisted), // custom merge
        }
      )
    ),
    { name: 'AppStore' } // devtools display name
  )
)

// External subscription with selector
useStore.subscribe(
  (state) => state.nested.deep.value,
  (value, prevValue) => console.log('changed', prevValue, '->', value)
)

// Testing: access state outside React
test('updateDeep increments', () => {
  const { updateDeep } = useStore.getState()
  act(() => updateDeep())
  expect(useStore.getState().nested.deep.value).toBe(1)
})
```

## Redux Toolkit with TypeScript

```typescript
export const store = configureStore({
  reducer: { user: userReducer, cart: cartReducer },
})
export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch
export const useAppDispatch: () => AppDispatch = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
```

```typescript
// Slice with async thunk
export const fetchUser = createAsyncThunk('user/fetchUser',
  async (userId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/users/${userId}`)
      if (!response.ok) throw new Error('Failed')
      return await response.json()
    } catch (error) { return rejectWithValue((error as Error).message) }
  }
)

const userSlice = createSlice({
  name: 'user', initialState,
  reducers: {
    setUser: (state, action: PayloadAction<User>) => { state.current = action.payload },
    clearUser: (state) => { state.current = null },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchUser.pending, (state) => { state.status = 'loading' })
      .addCase(fetchUser.fulfilled, (state, action) => { state.current = action.payload })
      .addCase(fetchUser.rejected, (state, action) => { state.error = action.payload as string })
  },
})
```

## Jotai for Atomic State

```typescript
import { atom } from 'jotai'
import { atomWithStorage } from 'jotai/utils'

export const userAtom = atom<User | null>(null)
export const isAuthenticatedAtom = atom((get) => get(userAtom) !== null)  // Derived
export const themeAtom = atomWithStorage<'light' | 'dark'>('theme', 'light')  // Persisted

// Async atom
export const userProfileAtom = atom(async (get) => {
  const user = get(userAtom)
  if (!user) return null
  return (await fetch(`/api/users/${user.id}/profile`)).json()
})

// Write-only action atom
export const logoutAtom = atom(null, (get, set) => {
  set(userAtom, null)
  set(cartAtom, [])
})
```

## React Query for Server State

```typescript
// Query keys factory
export const userKeys = {
  all: ['users'] as const,
  lists: () => [...userKeys.all, 'list'] as const,
  list: (filters: UserFilters) => [...userKeys.lists(), filters] as const,
  detail: (id: string) => [...userKeys.all, 'detail', id] as const,
}

// Mutation with optimistic update
export function useUpdateUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: updateUser,
    onMutate: async (newUser) => {
      await queryClient.cancelQueries({ queryKey: userKeys.detail(newUser.id) })
      const previousUser = queryClient.getQueryData(userKeys.detail(newUser.id))
      queryClient.setQueryData(userKeys.detail(newUser.id), newUser)
      return { previousUser }
    },
    onError: (err, newUser, context) => queryClient.setQueryData(userKeys.detail(newUser.id), context?.previousUser),
    onSettled: (data, error, variables) => queryClient.invalidateQueries({ queryKey: userKeys.detail(variables.id) }),
  })
}
```

## Combining Client + Server State

```typescript
// Zustand for UI state, React Query for server state
const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  modal: null,
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  openModal: (modal) => set({ modal }),
  closeModal: () => set({ modal: null }),
}))

function Dashboard() {
  const { sidebarOpen } = useUIStore()
  const { data: users, isLoading } = useUsers({ active: true })
  // ...
}
```

## Legacy Redux to RTK Migration

```typescript
// Before: action types, action creators, switch reducers
// After:
const todosSlice = createSlice({
  name: 'todos', initialState: [],
  reducers: {
    addTodo: (state, action: PayloadAction<string>) => {
      state.push({ text: action.payload, completed: false })  // Immer allows "mutations"
    },
  },
})
```

## Cross-References

- **frontend:nextjs-app-router-patterns** -- server/client state boundary, React Server Components context
- **frontend:form-patterns** -- form state management, React Hook Form integration
- **frontend:i18n-and-localization** -- locale state management, context providers
