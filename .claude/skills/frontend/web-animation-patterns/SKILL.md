---
name: web-animation-patterns
description: "Use when adding animations, transitions, or motion to web UIs. Covers library selection, CSS animations, Framer Motion, View Transitions API, FLIP technique, and reduced-motion accessibility."
---

# Web Animation Patterns

## Library Selection

| Need | Use |
|------|-----|
| Simple hover/toggle transitions | CSS transitions/animations |
| React layout + mount/unmount | Framer Motion |
| Complex sequenced timelines | GSAP |
| Page/route transitions | View Transitions API |
| Fine-grained imperative control | Web Animations API |

```
Performance-critical, simple      -> CSS only (transform, opacity)
React component animations        -> Framer Motion
Scroll-driven or complex sequence -> GSAP
Cross-document navigation         -> View Transitions API
```

## CSS Animations

```css
/* Compositor-only properties = 60fps */
.fade-in {
  animation: fadeIn 300ms ease-out forwards;
  will-change: opacity, transform; /* hint before animation starts, remove after */
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Spring-like easing via cubic-bezier */
.bounce { transition: transform 500ms cubic-bezier(0.34, 1.56, 0.64, 1); }
```

## Framer Motion

```tsx
import { motion, AnimatePresence } from 'framer-motion'

// Variants for orchestrated animations
const list = { hidden: {}, visible: { transition: { staggerChildren: 0.05 } } }
const item = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
}

function ItemList({ items }: { items: Item[] }) {
  return (
    <motion.ul variants={list} initial="hidden" animate="visible">
      {items.map((i) => (
        <motion.li key={i.id} variants={item} layout>
          {i.name}
        </motion.li>
      ))}
    </motion.ul>
  )
}

// AnimatePresence for exit animations
function Modal({ isOpen, children }: { isOpen: boolean; children: React.ReactNode }) {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
        >
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
```

## View Transitions API

```typescript
// SPA route transition
function navigate(href: string) {
  if (!document.startViewTransition) {
    updateDOM(href)
    return
  }
  document.startViewTransition(() => updateDOM(href))
}

// Named elements persist across transitions
// CSS: .card { view-transition-name: card-1; }
```

## FLIP Technique

First, Last, Invert, Play -- animate layout changes at 60fps.

```typescript
function flipAnimate(el: HTMLElement, update: () => void) {
  const first = el.getBoundingClientRect()    // First
  update()                                     // DOM change
  const last = el.getBoundingClientRect()      // Last
  const dx = first.left - last.left            // Invert
  const dy = first.top - last.top
  el.animate([
    { transform: `translate(${dx}px, ${dy}px)` },
    { transform: 'translate(0, 0)' },
  ], { duration: 300, easing: 'ease-out' })    // Play
}
```

## Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* Utility classes */
.motion-safe\:animate-fade { /* only in motion-safe context */ }
```

```typescript
// Framer Motion respects this automatically, or override:
const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
```

## Gotchas

- **Layout thrashing**: batch reads before writes; never interleave `getBoundingClientRect()` with DOM mutations
- **Compositor-only**: only `transform` and `opacity` avoid layout/paint -- animating `width`, `height`, `top`, `left` is expensive
- **`will-change`**: add before animation, remove after; permanent `will-change` wastes GPU memory
- **Framer Motion bundle**: ~30KB min+gz; use `LazyMotion` + `domAnimation` feature for code splitting
- **AnimatePresence**: requires direct children with stable `key` props; fragments break exit animations
- **View Transitions**: still limited cross-browser; always feature-detect with `document.startViewTransition`

## Cross-References

- **frontend:responsive-web-design** -- media queries and layout context for animation breakpoints
- **frontend:accessibility-testing** -- testing reduced-motion compliance and focus management
