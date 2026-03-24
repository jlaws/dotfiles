# Design Audit Checklist

Structured checklist for evaluating UI quality across dimensions. Optionally score each 0-10.

## Categories

### Typography
- Hierarchy: h1-h6 visually distinct and consistent
- Font scale: consistent ratio (e.g., 1.25 major second)
- Line height: 1.4-1.6 for body text
- Line width: 45-75 characters max for readability
- Responsive font sizing (clamp or fluid type)

### Spacing
- Consistent spacing scale (4px or 8px base)
- Adequate whitespace between sections
- Alignment grid respected across components
- Padding/margin consistency for similar components

### Color
- Design token usage — no hardcoded hex values
- Contrast: WCAG AA (4.5:1 text, 3:1 large text/UI)
- Semantic colors consistent (error=red, success=green, warning=amber)
- Dark mode / theme support via semantic tokens

### Visual Hierarchy
- Clear primary action per view (one dominant CTA)
- Information density appropriate for context
- Progressive disclosure for complex content
- Visual weight guides eye flow (size, color, position)

### Responsive
- Breakpoint behavior defined and tested
- Touch targets ≥44px on mobile
- No horizontal scroll at any breakpoint
- Images/media scale appropriately
- Navigation adapts (hamburger, tab bar, sidebar)

### Loading States
- Skeleton screens preferred over spinners
- Progress indicators for operations >2s
- Optimistic UI where safe (undo available)
- No content layout shift on load (CLS ≈ 0)

### Error States
- User-friendly messages (no stack traces, error codes)
- Recovery actions provided ("retry", "go back", "contact support")
- Form validation inline + on submit
- Network error handling with offline state

### Empty States
- Helpful messaging (not just "No data")
- Call-to-action to populate the view
- Illustration/icon appropriate and intentional

### Micro-interactions
- Hover/focus/active states defined for interactive elements
- Transitions smooth: 150-300ms, ease-out
- Feedback for user actions (click, submit, toggle)
- No jarring layout shifts during interaction

## "AI Slop" Detection

Red flags indicating AI-generated or low-effort design:

| Signal | Example | Fix |
|--------|---------|-----|
| Generic stock illustrations | Undraw/Storyset defaults unchanged | Custom illustration or remove |
| Placeholder copy | "Lorem ipsum", "Your amazing feature" | Real content, even if draft |
| Overly symmetric layouts | Perfect 3-column grid everywhere | Vary layout based on content hierarchy |
| Excessive gradients/shadows | Every card has drop shadow + gradient | Use sparingly for emphasis only |
| Default component styling | Unstyled Material/Chakra/Shadcn defaults | Apply design tokens and brand |
