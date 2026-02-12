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

```typescript
// app/products/page.tsx
export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>
}) {
  const params = await searchParams
  return (
    <div className="flex gap-8">
      <FilterSidebar />
      <Suspense key={JSON.stringify(params)} fallback={<ProductListSkeleton />}>
        <ProductList category={params.category} sort={params.sort} page={Number(params.page) || 1} />
      </Suspense>
    </div>
  )
}

// Server Component fetches its own data
async function ProductList({ category, sort, page }: ProductFilters) {
  const { products, totalPages } = await getProducts({ category, sort, page })
  return (
    <div>
      <div className="grid grid-cols-3 gap-4">
        {products.map((product) => <ProductCard key={product.id} product={product} />)}
      </div>
      <Pagination currentPage={page} totalPages={totalPages} />
    </div>
  )
}
```

## Pattern 2: Server Actions

```typescript
// app/actions/cart.ts
'use server'

import { revalidateTag } from 'next/cache'
import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'

export async function addToCart(productId: string) {
  const cookieStore = await cookies()
  const sessionId = cookieStore.get('session')?.value
  if (!sessionId) redirect('/login')

  try {
    await db.cart.upsert({
      where: { sessionId_productId: { sessionId, productId } },
      update: { quantity: { increment: 1 } },
      create: { sessionId, productId, quantity: 1 },
    })
    revalidateTag('cart')
    return { success: true }
  } catch (error) {
    return { error: 'Failed to add item to cart' }
  }
}
```

## Pattern 3: Parallel Routes

```typescript
// app/dashboard/layout.tsx
export default function DashboardLayout({
  children, analytics, team,
}: {
  children: React.ReactNode
  analytics: React.ReactNode
  team: React.ReactNode
}) {
  return (
    <div className="dashboard-grid">
      <main>{children}</main>
      <aside>{analytics}</aside>
      <aside>{team}</aside>
    </div>
  )
}
// app/dashboard/@analytics/page.tsx, @analytics/loading.tsx, @team/page.tsx
```

## Pattern 4: Intercepting Routes (Modal Pattern)

```typescript
// app/@modal/(.)photos/[id]/page.tsx - Intercept for modal
// app/photos/[id]/page.tsx - Full page version
// app/layout.tsx receives both {children} and {modal}
```

## Pattern 5: Streaming with Suspense

```typescript
export default async function ProductPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const product = await getProduct(id) // Blocking

  return (
    <div>
      <ProductHeader product={product} />
      <Suspense fallback={<ReviewsSkeleton />}>
        <Reviews productId={id} />  {/* Streams in */}
      </Suspense>
      <Suspense fallback={<RecommendationsSkeleton />}>
        <Recommendations productId={id} />  {/* Streams in */}
      </Suspense>
    </div>
  )
}
```

## Pattern 6: Metadata and SEO

```typescript
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const product = await getProduct(slug)
  if (!product) return {}
  return {
    title: product.name,
    description: product.description,
    openGraph: { title: product.name, images: [{ url: product.image, width: 1200, height: 630 }] },
  }
}

export async function generateStaticParams() {
  const products = await db.product.findMany({ select: { slug: true } })
  return products.map((p) => ({ slug: p.slug }))
}
```

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

// Invalidate via Server Action
'use server'
import { revalidateTag, revalidatePath } from 'next/cache'
export async function updateProduct(id: string, data: ProductData) {
  await db.product.update({ where: { id }, data })
  revalidateTag('products')
  revalidatePath('/products')
}
```

## Route Handlers (API Routes)

```typescript
// app/api/products/route.ts
export async function GET(request: NextRequest) {
  const category = request.nextUrl.searchParams.get('category')
  const products = await db.product.findMany({
    where: category ? { category } : undefined, take: 20,
  })
  return NextResponse.json(products)
}

export async function POST(request: NextRequest) {
  const body = await request.json()
  const product = await db.product.create({ data: body })
  return NextResponse.json(product, { status: 201 })
}
```

## Pattern 7: Forms with useActionState

```typescript
'use client'
import { useActionState } from 'react'
import { createUser } from '@/app/actions/user'

export function CreateUserForm() {
  const [state, formAction, isPending] = useActionState(createUser, { errors: {} })
  return (
    <form action={formAction}>
      <input name="email" />
      {state.errors?.email && <p>{state.errors.email}</p>}
      <button disabled={isPending}>{isPending ? 'Creating...' : 'Create'}</button>
    </form>
  )
}
```

## Pattern 8: Middleware

```typescript
// middleware.ts
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export function middleware(request: NextRequest) {
  const token = request.cookies.get('token')?.value
  if (!token && request.nextUrl.pathname.startsWith('/dashboard')) {
    return NextResponse.redirect(new URL('/login', request.url))
  }
  return NextResponse.next()
}

export const config = { matcher: ['/dashboard/:path*', '/api/protected/:path*'] }
```

## Cross-References

- **frontend:form-patterns** -- React Hook Form, Zod validation, multi-step forms
- **frontend:react-state-management** -- client state patterns, Zustand/Jotai with Next.js
- **frontend:i18n-and-localization** -- next-intl setup, locale routing middleware
