
# Web Animation Patterns

## Library Selection

| Need | Use |
|------|-----|
| Simple hover/toggle transitions | CSS transitions/animations |
| React layout + mount/unmount | Framer Motion |
| Complex sequenced timelines | GSAP |
| Page/route transitions | View Transitions API |
| Fine-grained imperative control | Web Animations API |

## CSS Animations

Use `transform` and `opacity` only (compositor-only, 60fps). Add `will-change` before animation, remove after. Use `cubic-bezier()` for spring-like easing.

## Framer Motion

Use `variants` for orchestrated animations with `staggerChildren`. `AnimatePresence` wraps conditional renders for exit animations. `layout` prop animates layout changes. `LazyMotion` + `domAnimation` for code splitting (~30KB savings).

## View Transitions API

Call `document.startViewTransition(() => updateDOM())` with feature detection fallback. Use `view-transition-name` CSS for element persistence across transitions.

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

Reduced motion means fewer and gentler animations, not none. Remove translation, scaling, parallax, and autoplaying loops. Keep short opacity and color changes that confirm an action or show state. A global `0.01ms` duration kill erases that feedback too.

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    /* transforms and size changes jump; fades and color changes keep their timing */
    transition-property: opacity, color, background-color, border-color, box-shadow !important;
  }
  /* Spatial motion: run to the final keyframe at once, so fill-mode: forwards entrances end visible */
  [data-motion="spatial"] {
    animation-duration: 1ms !important;
    animation-iteration-count: 1 !important;
  }
}
```

```typescript
// Motion ignores this by default (MotionConfig reducedMotion="never"); opt in with <MotionConfig reducedMotion="user">, or read it directly:
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
