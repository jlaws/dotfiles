# Design Audit Checklist

Structured checklist for evaluating UI quality across dimensions. Optionally score each 0-10.

## Categories

### Visual Hierarchy
- Clear primary action per view (one dominant CTA)

### Loading States
- Skeleton screens preferred over spinners
- Progress indicators for operations >2s
- Optimistic UI where safe (undo available)

## "AI Slop" Detection

Red flags indicating AI-generated or low-effort design:

| Signal | Example | Fix |
|--------|---------|-----|
| Generic stock illustrations | Undraw/Storyset defaults unchanged | Custom illustration or remove |
| Placeholder copy | "Lorem ipsum", "Your amazing feature" | Real content, even if draft |
| Overly symmetric layouts | Perfect 3-column grid everywhere | Vary layout based on content hierarchy |
| Excessive gradients/shadows | Every card has drop shadow + gradient | Use sparingly for emphasis only |
| Default component styling | Unstyled Material/Chakra/Shadcn defaults | Apply design tokens and brand |
