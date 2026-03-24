
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

Base styles = mobile. Layer up with `min-width` breakpoints: 640px (sm), 1024px (md), 1280px (lg).

## Responsive Images

Use `srcset` + `sizes` for resolution switching, `<picture>` + `<source media>` for art direction. Always `loading="lazy"` below fold.

### Code Example: srcset, sizes, and Picture Element

```html
<!-- Resolution switching: browser picks 1x or 2x density -->
<img
  src="image-320w.jpg"
  srcset="image-320w.jpg 320w, image-640w.jpg 640w, image-1200w.jpg 1200w"
  sizes="(max-width: 640px) 100vw, (max-width: 1200px) 50vw, 33vw"
  alt="Responsive example"
  loading="lazy"
/>

<!-- Art direction: different crops for different viewports -->
<picture>
  <source media="(max-width: 640px)" srcset="hero-mobile.jpg">
  <source media="(max-width: 1200px)" srcset="hero-tablet.jpg">
  <img src="hero-desktop.jpg" alt="Hero banner" loading="lazy" />
</picture>

<!-- WebP with fallback -->
<picture>
  <source srcset="image.webp" type="image/webp">
  <source srcset="image.jpg" type="image/jpeg">
  <img src="image.jpg" alt="Description" loading="lazy" />
</picture>
```

## CSS Grid Patterns

Use `repeat(auto-fit, minmax(250px, 1fr))` for breakpoint-free responsive grids. Sidebar: fixed + `minmax(0, 1fr)` behind a media query.

### Code Example: Responsive Grid Layout

```css
/* Auto-fit grid: automatically reflows based on space */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

/* Sidebar + main content: stack on mobile, side-by-side on desktop */
.layout {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1rem;
}

@media (min-width: 1024px) {
  .layout {
    grid-template-columns: 250px 1fr;
  }

  .sidebar {
    position: sticky;
    top: 1rem;
    height: fit-content;
  }
}

/* 3-column grid with controlled column width */
.product-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  auto-rows: 300px;
  gap: 1.5rem;
}

/* Named areas for flexible layouts */
.dashboard {
  display: grid;
  grid-template-columns: 1fr;
  grid-template-areas:
    'header'
    'sidebar'
    'main'
    'footer';
}

@media (min-width: 1024px) {
  .dashboard {
    grid-template-columns: 250px 1fr;
    grid-template-areas:
      'header header'
      'sidebar main'
      'footer footer';
  }
}
```

## Viewport Units

Dynamic viewport height units handle mobile browser chrome (address bar, bottom nav).

### Code Example: Viewport Units (dvh, svh, lvh)

```css
/* Dynamic viewport height: adjusts for browser chrome
   Use for full-screen sections that need to breathe with mobile UI */
.hero {
  height: 100dvh; /* Adjusts when address bar appears/disappears */
}

/* Small viewport height: excludes keyboard and bottom nav
   Use when you want consistent layout despite mobile chrome */
.input-modal {
  max-height: 100svh;
  overflow-y: auto;
}

/* Large viewport height: largest possible viewport
   Use for less responsive cases where you want max space */
.fullscreen-map {
  height: 100lvh;
}

/* Safe fallback for older browsers */
@supports not (height: 100dvh) {
  .hero {
    height: 100vh;
    height: 100dvh;
  }
}

/* Practical: sticky header + dynamic viewport */
body {
  display: flex;
  flex-direction: column;
  height: 100dvh;
}

.header {
  flex-shrink: 0;
  height: 60px;
}

.main {
  flex: 1;
  overflow-y: auto;
}
```

## Performance & Core Web Vitals

### Font Loading & CLS (Cumulative Layout Shift)

```css
/* font-display: swap prevents invisible text during load
   size-adjust reduces baseline shift from font swap */
@font-face {
  font-family: 'CustomFont';
  src: url('font.woff2') format('woff2');
  font-display: swap;
  size-adjust: 100%;
}

/* Reserve space for font-loaded text to prevent shift */
.heading {
  font-family: 'CustomFont', serif;
  line-height: 1.4;
  min-height: 1.4em;
}

/* Avoid margin shifts: use padding on parent or explicit height */
.text-skeleton {
  height: 1.5em;
  background: linear-gradient(90deg, #eee 25%, #f3f3f3 50%, #eee 75%);
  background-size: 200% 100%;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
```

### Image Optimization

```html
<!-- Lazy loading with proper dimensions to prevent layout shift -->
<img
  src="image.jpg"
  loading="lazy"
  width="400"
  height="300"
  alt="Description"
  style="aspect-ratio: 4 / 3;"
/>

<!-- Native aspect-ratio prevents CLS -->
<img
  src="image.jpg"
  alt="Description"
  style="aspect-ratio: 16 / 9; width: 100%; height: auto;"
/>

<!-- Blurred placeholder during load (LQIP pattern) -->
<img
  src="image-small.jpg"
  loading="lazy"
  alt="High-res image"
  style="filter: blur(5px);"
  onload="this.style.filter = 'none';"
/>
```

## Performance Budget Targets

| Metric | Target | Measurement |
|---|---|---|
| **LCP** (Largest Contentful Paint) | <2.5s | Core Web Vital — largest visible element render time |
| **FID** (First Input Delay) | <100ms | Core Web Vital — time to first interaction response |
| **INP** (Interaction to Next Paint) | <200ms | Core Web Vital (replaces FID March 2024) |
| **CLS** (Cumulative Layout Shift) | <0.1 | Core Web Vital — visual stability score |
| **Lighthouse Performance** | 90+ | Synthetic benchmark floor |
| **Page load (3G)** | <3s | Mobile constraint — test with Chrome DevTools throttling |
| **Total page weight** | <500KB initial | First load; lazy-load the rest |
| **JavaScript bundle** | <150KB gzipped | Main bundle; code-split aggressively |

### Enforcement

- Run Lighthouse CI in pull requests — block merge if score drops below 85
- Set `performance.budgets` in `next.config.js` or webpack config
- Track Core Web Vitals in production via `web-vitals` library → analytics pipeline

## Gotchas

- **Viewport units on mobile**: `100vh` includes browser chrome; use `100dvh` (dynamic viewport height) instead
- **Container query support**: Baseline 2023 -- add `@supports (container-type: inline-size)` fallback for older browsers
- **Touch targets**: Minimum 44x44px (WCAG 2.5.5); use `min-height: 44px; min-width: 44px` on interactive elements
- **Horizontal scroll**: Always test with `overflow-x: hidden` on body during dev to catch overflow; use `max-width: 100%` on images/videos
- **Font loading shift**: Use `font-display: swap` and `size-adjust` to minimize CLS from web font loading
- **Grid auto-fit vs auto-fill**: `auto-fit` collapses empty tracks; `auto-fill` keeps them (subtle perf difference)
- **Aspect ratio with srcset**: Always declare `aspect-ratio` CSS to prevent layout shift before image loads

## Cross-References

- **frontend:tailwind-design-system** -- Tailwind responsive utilities, breakpoint config, container queries plugin
- **frontend:accessibility-testing** -- Touch target compliance, responsive WCAG requirements
