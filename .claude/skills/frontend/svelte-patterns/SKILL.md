---
name: svelte-patterns
description: "Use when building Svelte 5 or SvelteKit applications. Covers runes, SvelteKit routing and SSR, component patterns, stores, transitions, and migration from Svelte 4."
---

# Svelte Patterns

## Svelte 5 Runes

Runes replace Svelte 4's `let` reactivity and `$:` labels with explicit primitives.

```svelte
<script lang="ts">
  // Reactive state
  let count = $state(0)
  let items = $state<string[]>([])

  // Derived values (replaces $: derived = ...)
  let doubled = $derived(count * 2)
  let total = $derived.by(() => {
    return items.reduce((sum, item) => sum + item.length, 0)
  })

  // Side effects (replaces $: { ... })
  $effect(() => {
    console.log(`count is ${count}`)
    return () => console.log('cleanup')  // cleanup function
  })
</script>

<button onclick={() => count++}>{count} (doubled: {doubled})</button>
```

## Component Props (Svelte 5)

```svelte
<script lang="ts">
  // $props replaces export let
  interface Props {
    title: string
    count?: number
    children: import('svelte').Snippet  // replaces <slot>
    header?: import('svelte').Snippet<[string]>  // named snippet with args
  }
  let { title, count = 0, children, header }: Props = $props()
</script>

<div>
  {#if header}{@render header(title)}{/if}
  {@render children()}
</div>
```

## SvelteKit Routing

```
src/routes/
  +page.svelte          # /
  +layout.svelte        # shared layout
  +page.server.ts       # server load function
  blog/
    +page.svelte        # /blog
    [slug]/
      +page.svelte      # /blog/:slug
      +page.ts           # universal load
      +page.server.ts    # server-only load
  api/
    posts/
      +server.ts         # API route: GET, POST, etc.
```

### Load Functions

```typescript
// +page.server.ts (server-only, access DB/secrets)
import type { PageServerLoad } from './$types'

export const load: PageServerLoad = async ({ params, locals }) => {
  const post = await db.posts.findUnique({ where: { slug: params.slug } })
  if (!post) throw error(404, 'Not found')
  return { post }  // typed, available as `data` prop
}

// +page.ts (universal, runs server + client)
import type { PageLoad } from './$types'

export const load: PageLoad = async ({ fetch, params }) => {
  const res = await fetch(`/api/posts/${params.slug}`)
  return { post: await res.json() }
}
```

### Form Actions

```typescript
// +page.server.ts
import { fail, redirect } from '@sveltejs/kit'
import type { Actions } from './$types'

export const actions: Actions = {
  create: async ({ request }) => {
    const data = await request.formData()
    const title = data.get('title')
    if (!title) return fail(400, { title, missing: true })
    await db.posts.create({ data: { title: String(title) } })
    throw redirect(303, '/posts')
  },
}
```

```svelte
<!-- +page.svelte -->
<script lang="ts">
  import { enhance } from '$app/forms'
  let { form } = $props()  // form action return data
</script>

<form method="POST" action="?/create" use:enhance>
  <input name="title" value={form?.title ?? ''} />
  {#if form?.missing}<p>Title is required</p>{/if}
  <button>Create</button>
</form>
```

## Stores (Svelte 4/5 Compatible)

```typescript
import { writable, derived, readable } from 'svelte/store'

// Writable store
export const user = writable<User | null>(null)

// Derived store
export const isLoggedIn = derived(user, ($user) => $user !== null)

// Custom store with methods
function createCounter() {
  const { subscribe, set, update } = writable(0)
  return {
    subscribe,
    increment: () => update((n) => n + 1),
    reset: () => set(0),
  }
}
export const counter = createCounter()
```

Access in components with `$` prefix: `$user`, `$isLoggedIn`, `$counter`.

## Transitions

```svelte
<script>
  import { fade, fly, slide, scale } from 'svelte/transition'
  import { flip } from 'svelte/animate'
  let visible = $state(true)
</script>

{#if visible}
  <div transition:fade={{ duration: 300 }}>fades in and out</div>
  <div in:fly={{ y: 20 }} out:fade>different in/out</div>
{/if}

<!-- Animate list reordering -->
{#each items as item (item.id)}
  <div animate:flip={{ duration: 200 }} transition:slide>
    {item.name}
  </div>
{/each}
```

## Gotchas

- **Svelte 5 migration**: `let x = 0` is no longer reactive at top level -- must use `$state(0)`. `$:` labels are deprecated; use `$derived` and `$effect`
- **`$effect` vs `$derived`**: use `$derived` for computed values, `$effect` only for side effects. `$effect` does NOT return a value
- **Object/array reactivity**: `$state` uses proxies; direct property mutation works (`obj.key = val`), but reassigning nested arrays needs care: `items = [...items, newItem]` or mutate in place
- **Stores to runes**: Svelte 5 still supports stores but runes are preferred. `$state` replaces `writable` for component-local state; stores still useful for cross-component shared state
- **Hydration mismatch**: SvelteKit SSR means server/client must render identically. Guard browser APIs with `browser` from `$app/environment` or `onMount`
- **Form action gotcha**: `throw redirect()` (not `return redirect()`) -- redirect is implemented as a thrown response in SvelteKit
- **`use:enhance`**: without it, form submissions trigger full page reload. With it, SvelteKit progressively enhances to fetch

## Cross-References

- **frontend:form-patterns** -- advanced form validation and multi-step wizard patterns
- **frontend:react-state-management** -- compare Svelte stores/runes with React state solutions
