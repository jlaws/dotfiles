---
name: premium-design-aesthetics
description: "Opinionated rules for premium UI design — banned fonts, color constraints, layout anti-patterns, and AI fingerprints to eliminate. Load when building or auditing UIs for quality."
type: reference
---

# Premium Frontend Design Aesthetics

## Typography

| Rule | Detail |
|------|--------|
| **Banned for premium contexts** | Inter, Roboto — too generic, AI default fingerprint |
| **Preferred fonts** | Geist, Outfit, Cabinet Grotesk, Satoshi |
| **Body text** | Off-black only: `#111111` or `#2F3437` — never pure `#000000` |
| **Weight discipline** | Limit to 2 weights per component; avoid weight mixing for hierarchy |

## Color

| Rule | Detail |
|------|--------|
| **Accent saturation** | Single accent color, <80% saturation |
| **Backgrounds** | Warm neutrals: `#F9FAFB` (off-white) or `#FFFFFF`; never pure gray |
| **Body text** | `#111111` or `#2F3437` — off-black, not pure black |
| **Banned palettes** | Purple/neon gradients (AI fingerprint), equal blue/purple duotones |
| **Gradient discipline** | Subtle background gradients OK; gradient cards and gradient buttons = AI fingerprint |

## Layout Anti-Patterns

| Anti-pattern | Replace with |
|---|---|
| Centered hero with large headline + subtext + CTA | Off-center layouts with visual tension |
| Equal 3-column card grids | Asymmetric grids or single-column with varying widths |
| Overlapping elements without intentional layering | Clean z-index hierarchy or flat layout |
| Missing mobile collapse | Mandatory single-column collapse below 768px |

## Materiality & Spacing

- **Spacing over cards**: Use whitespace and borders for content separation before reaching for card components
- **No shadow stacking**: One shadow level per elevation; nested shadows = visual noise
- **Border discipline**: 1px borders at `opacity: 0.08–0.12` for subtle separation

## Interactive States

All interactive components **must implement** all four states. Omitting any = broken component.

| State | Requirement |
|-------|-------------|
| Loading | Skeleton or spinner — never blank |
| Empty | Illustrated empty state with CTA |
| Error | Inline error with recovery action |
| Success | Confirmation feedback (toast, inline, or animation) |

## Animation

| Rule | Detail |
|------|--------|
| **Physics model** | Spring animations only: `stiffness: 100, damping: 20` |
| **GPU-safe properties** | Only animate `transform` and `opacity` — never `width`, `height`, `top`, `left` |
| **Duration** | 150–300ms for micro-interactions; 400–600ms for page transitions |
| **Banned** | Circular spinners (AI fingerprint), bounce easing on non-playful UIs, CSS `transition: all` |

## AI Fingerprints to Eliminate

These patterns are signals that a UI was AI-generated without design oversight:

- [ ] Inter or Roboto as primary font
- [ ] Gradient cards or gradient CTA buttons
- [ ] Equal 3-column feature grid
- [ ] Missing hover/focus states
- [ ] Purple/blue duotone color palette
- [ ] Circular spinner as only loading state
- [ ] Emojis in UI copy (outside chat/social contexts)
- [ ] Round numbers in "social proof" (e.g., "10,000+ users", "99% uptime")
- [ ] Pure black (`#000000`) body text
- [ ] Centered hero with identical padding on all sides

## Audit Checklist

Run this before shipping any UI:

1. Font: Is it Geist/Outfit/Cabinet Grotesk/Satoshi or equivalent?
2. Color: Single accent, <80% saturation, warm neutrals?
3. Text: Off-black (`#111`/`#2F3437`), never pure black?
4. Layout: No equal 3-col grid? Mobile collapses correctly?
5. States: All 4 states (loading, empty, error, success) implemented?
6. Animation: Spring physics, GPU-safe properties only?
7. AI fingerprints: Zero items from checklist above?

## Cross-References

- **frontend:design-system-patterns** — token hierarchy, component APIs, headless libraries
- **frontend:tailwind-design-system** — Tailwind-specific implementation of these constraints
- **frontend:web-animation-patterns** — spring animation implementation with Framer Motion / CSS
