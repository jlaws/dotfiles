---
name: responsive-web-design
description: "Use when implementing responsive layouts, fluid typography, responsive images, or choosing between container queries and media queries."
---

# Responsive Web Design

## Container Queries vs Media Queries

| Feature | Media Queries | Container Queries |
|---|---|---|
| Based on | Viewport size | Parent container size |
| Use case | Page-level layout | Component-level layout |
| Nesting | N/A | Supported |
| Browser support | Universal | Modern browsers (2023+) |

**Default**: Media queries for page layout (nav, sidebar). Container queries for reusable components that render in different contexts.

```css
/* Container query setup */
.card-wrapper {
  container-type: inline-size;
  container-name: card;
}

@container card (min-width: 400px) {
  .card { flex-direction: row; }
  .card-image { width: 40%; }
}

@container card (max-width: 399px) {
  .card { flex-direction: column; }
  .card-image { width: 100%; }
}
```

## Fluid Typography

```css
/* clamp(min, preferred, max) — no breakpoints needed */
:root {
  --text-sm: clamp(0.8rem, 0.17vw + 0.76rem, 0.89rem);
  --text-base: clamp(1rem, 0.34vw + 0.91rem, 1.19rem);
  --text-lg: clamp(1.25rem, 0.61vw + 1.1rem, 1.58rem);
  --text-xl: clamp(1.56rem, 1vw + 1.31rem, 2.11rem);
  --text-2xl: clamp(1.95rem, 1.56vw + 1.56rem, 2.81rem);
  --text-3xl: clamp(2.44rem, 2.38vw + 1.85rem, 3.75rem);
}

h1 { font-size: var(--text-3xl); }
p { font-size: var(--text-base); }
```

## Mobile-First Strategy

```css
/* Mobile-first: base styles = mobile, layer up with min-width */
.grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

@media (min-width: 640px)  { .grid { grid-template-columns: repeat(2, 1fr); } }
@media (min-width: 1024px) { .grid { grid-template-columns: repeat(3, 1fr); } }
@media (min-width: 1280px) { .grid { grid-template-columns: repeat(4, 1fr); gap: 1.5rem; } }
```

## Responsive Images

```html
<!-- srcset + sizes: browser picks optimal source -->
<img
  srcset="hero-400.webp 400w, hero-800.webp 800w, hero-1200.webp 1200w"
  sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
  src="hero-800.webp"
  alt="Hero image"
  loading="lazy"
  decoding="async"
/>

<!-- Art direction: different crops per breakpoint -->
<picture>
  <source media="(min-width: 1024px)" srcset="hero-wide.webp" />
  <source media="(min-width: 640px)" srcset="hero-medium.webp" />
  <img src="hero-square.webp" alt="Hero" />
</picture>
```

```typescript
// Next.js responsive image
import Image from 'next/image'

<Image
  src="/hero.webp"
  alt="Hero"
  fill
  sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
  priority  // above-the-fold only
/>
```

## CSS Grid + Flexbox Patterns

```css
/* Auto-fit grid: items wrap responsively without breakpoints */
.auto-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(250px, 100%), 1fr));
  gap: 1.5rem;
}

/* Sidebar layout: sidebar fixed, main fluid */
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
}
@media (min-width: 768px) {
  .layout { grid-template-columns: 250px minmax(0, 1fr); }
}

/* Flex wrap with minimum child width */
.flex-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}
.flex-wrap > * {
  flex: 1 1 300px; /* grow, shrink, min-width basis */
}
```

## Gotchas

- **Viewport units on mobile**: `100vh` includes browser chrome; use `100dvh` (dynamic viewport height) instead
- **Container query support**: Baseline 2023 -- add `@supports (container-type: inline-size)` fallback for older browsers
- **Touch targets**: Minimum 44x44px (WCAG 2.5.5); use `min-height: 44px; min-width: 44px` on interactive elements
- **Horizontal scroll**: Always test with `overflow-x: hidden` on body during dev to catch overflow; use `max-width: 100%` on images/videos
- **Font loading shift**: Use `font-display: swap` and `size-adjust` to minimize CLS from web font loading

## Cross-References

- **frontend:tailwind-design-system** -- Tailwind responsive utilities, breakpoint config, container queries plugin
- **frontend:accessibility-testing** -- Touch target compliance, responsive WCAG requirements
