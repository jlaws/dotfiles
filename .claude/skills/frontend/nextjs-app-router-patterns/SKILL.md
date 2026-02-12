---
name: nextjs-app-router-patterns
description: Master Next.js 15 App Router with Server Components, streaming, parallel routes, and advanced data fetching. Use when building Next.js applications, implementing SSR/SSG, or optimizing React Server Components.
---

# Next.js App Router Patterns

## Rendering Modes

| Mode | Where | When to Use |
|------|-------|-------------|
| **Server Components** | Server only | Data fetching, heavy computation, secrets |
| **Client Components** | Browser | Interactivity, hooks, browser APIs |
| **Static** | Build time | Content that rarely changes |
| **Dynamic** | Request time | Personalized or real-time data |
| **Streaming** | Progressive | Large pages, slow data sources |

## File Conventions

```
app/
├── layout.tsx       # Shared UI wrapper
├── page.tsx         # Route UI
├── loading.tsx      # Loading UI (Suspense)
├── error.tsx        # Error boundary
├── not-found.tsx    # 404 UI
├── route.ts         # API endpoint
├── template.tsx     # Re-mounted layout
├── default.tsx      # Parallel route fallback
└── opengraph-image.tsx  # OG image generation
```

## Pattern 1: Server Components with Data Fetching

Fetch data in async server components. Wrap slow data in `<Suspense>` with fallback. Use `key={JSON.stringify(searchParams)}` to reset Suspense on filter changes.

## Pattern 2: Server Actions

Mark files `'use server'`. Return structured `{ success }` or `{ error }` objects. Call `revalidateTag()`/`revalidatePath()` after mutations.

## Pattern 3: Parallel Routes

Use `@slot` directories. Layout receives slot props alongside `children`. Each slot gets its own loading/error boundaries.

## Pattern 4: Intercepting Routes

Use `(.)path` convention to intercept for modals while preserving direct-link full-page view.

## Pattern 5: Streaming with Suspense

Nest `<Suspense>` boundaries for independent data. Blocking data fetches above, streaming data below. Each boundary streams independently.

## Pattern 6: Metadata and SEO

Export async `generateMetadata()` for dynamic SEO. Export `generateStaticParams()` for SSG.

## Next.js 15 Breaking Changes

**Async Request APIs**: All request-time APIs are now async and must be awaited:
```typescript
// Next.js 15 -- cookies(), headers(), params, searchParams are all async
const cookieStore = await cookies()
const headersList = await headers()
const { id } = await params
const { query } = await searchParams
```

**Fetch caching**: `fetch()` is no longer cached by default in Next.js 15. You must opt in:
```typescript
// Next.js 14: cached by default
// Next.js 15: NOT cached by default -- must explicitly opt in
fetch(url, { cache: 'force-cache' })  // Opt in to caching
```

**Turbopack**: Stable for `next dev --turbo` in Next.js 15. 10x faster HMR, 4x faster cold starts. Not yet stable for production builds.

**`after()` API**: Run code after the response is sent (analytics, logging, cleanup):
```typescript
import { after } from 'next/server'

export async function POST(request: NextRequest) {
  const data = await processRequest(request)
  after(() => {
    // Runs after response is sent -- does not block the user
    analyticsTrack('api_call', { endpoint: '/api/process' })
  })
  return NextResponse.json(data)
}
```

**Partial Prerendering (PPR)**: Experimental. Combines static shell with streaming dynamic holes. Enable per-route with `experimental_ppr = true` in route segment config.

## Caching Strategies

```typescript
fetch(url, { cache: 'no-store' })              // Always fresh (Next.js 15 default)
fetch(url, { cache: 'force-cache' })            // Cache forever (must opt in)
fetch(url, { next: { revalidate: 60 } })        // ISR - 60 seconds
fetch(url, { next: { tags: ['products'] } })     // Tag-based invalidation
```

## Route Handlers

Export named HTTP method functions (`GET`, `POST`) from `route.ts` files.

## Middleware

Export `middleware()` from root `middleware.ts`. Use `config.matcher` array to scope.

## Cross-References

- **frontend:form-patterns** -- React Hook Form, Zod validation, multi-step forms
- **frontend:react-state-management** -- client state patterns, Zustand/Jotai with Next.js
- **frontend:i18n-and-localization** -- next-intl setup, locale routing middleware
