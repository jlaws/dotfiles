---
name: redesign
description: "Audit existing UI for AI fingerprints and generic patterns, then produce a priority-ranked redesign plan. Use when a UI looks AI-generated, needs premium polish, or has been flagged for generic aesthetics."
argument-hint: "<component, page, or directory to audit>"
---

Before starting, gather diagnostic context:

1. **Detect framework** from config files (next.config.js, vite.config.ts, package.json).
2. **Detect styling approach**: Tailwind config, CSS modules, styled-components, design token files.
3. **Identify fonts**: Grep for font imports in CSS, `_document.tsx`, `layout.tsx`, or `index.html`.
4. **Scan for AI fingerprints**: Search for `Inter`, `Roboto`, gradient classes, `grid-cols-3`, hardcoded `#000000`.
5. **Scope the target**: If `$ARGUMENTS` names a component/page, read it directly. Otherwise scan `src/`, `app/`, `pages/` for the largest/most visible UI surfaces.

Then proceed with the following redesign framework:

---

## Redesign Audit Framework

### Step 1: Scan — Detect AI Fingerprints

Check against `references/frontend/premium-design-aesthetics` fingerprint checklist:

- [ ] Font: Inter or Roboto in use?
- [ ] Color: Purple/neon gradients? Pure black text?
- [ ] Layout: Equal 3-column grid? Centered hero?
- [ ] States: Missing hover, loading, empty, or error states?
- [ ] Animation: Circular spinners? CSS `transition: all`?
- [ ] Copy: Emojis? Round social-proof numbers?

### Step 2: Diagnose — Priority-Ranked Fix List

Order fixes by impact (biggest visual improvement per line of code):

| Priority | Fix | Impact |
|----------|-----|--------|
| 1 | Font swap (Inter → Geist/Outfit/Satoshi) | High — changes entire UI character |
| 2 | Color cleanup (palette + text color) | High — eliminates generic feel |
| 3 | Interactive state implementation | High — removes broken/incomplete feel |
| 4 | Layout refinement (grid asymmetry, spacing) | Medium |
| 5 | Component replacement (cards → spacing, gradient → flat) | Medium |
| 6 | Animation upgrade (spring physics, GPU-safe) | Low-Medium |

### Step 3: Fix — Implementation Rules

- **Preserve existing stack**: Do not swap frameworks or CSS libraries
- **Biggest impact first**: Always fix fonts before colors, colors before layout
- **No new dependencies** unless the font or animation library is genuinely absent
- **Show before/after**: For every change, note what was replaced and why

---

Read `references/frontend/premium-design-aesthetics` and `references/frontend/design-system-patterns` before producing recommendations.

Target: $ARGUMENTS
