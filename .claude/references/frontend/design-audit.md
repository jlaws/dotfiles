# Design Audit Checklist

Structured checklist for evaluating UI quality across dimensions. Optionally score each 0-10.

## Categories

### Visual Hierarchy
- Clear primary action per view (one dominant CTA)

### Loading States
- Skeleton screens preferred over spinners
- Progress indicators for operations >2s
- Optimistic UI where safe (undo available)

## Triage and Severity

Fix in this order, and do not perfect one corner while the rest sits below the bar:

1. Broken or blocked tasks, data loss, misleading state, inaccessible paths
2. Missing loading, empty, error, success, disabled, and permission states
3. Flow, hierarchy, responsive, and design-system drift
4. Visual and motion inconsistencies
5. Code and asset cleanup

Classify each drift by where its fix belongs: a missing token (fix the token), a one-off implementation (move it into the shared component), a conceptual mismatch (fix the flow), or a local defect (fix in place). Judge the rendered result at shipping sizes, and form your own assessment before reading any scanner's output, which anchors judgment.

## "AI Slop" Detection

Red flags indicating AI-generated or low-effort design:

| Signal | Example | Fix |
|--------|---------|-----|
| Generic stock illustrations | Undraw/Storyset defaults unchanged | Custom illustration or remove |
| Placeholder copy | "Lorem ipsum", "Your amazing feature" | Real content, even if draft |
| Overly symmetric layouts | Perfect 3-column grid everywhere | Vary layout based on content hierarchy |
| Excessive gradients/shadows | Every card has drop shadow + gradient | Use sparingly for emphasis only |
| Default component styling | Unstyled Material/Chakra/Shadcn defaults | Apply design tokens and brand |
