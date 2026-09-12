# Completeness Principle

AI-assisted development compresses the cost of thoroughness. When Claude Code effort is low relative to human effort, prefer thorough over "good enough." The marginal cost of completeness is near-zero for many tasks that would take humans significantly longer.

## Dual Effort Estimation

| Task | Human Effort | CC Effort | Verdict |
|------|-------------|-----------|---------|
| Full edge-case test coverage | 2-4 hours | 5-10 min | Do it |
| Input validation for all paths | 1-2 hours | 3-5 min | Do it |
| Error messages with context | 1-2 hours | 2-5 min | Do it |
| Inline documentation for complex logic | 30-60 min | 2-3 min | Do it |
| Migration script with rollback | 2-4 hours | 10-15 min | Do it |
| New feature not in spec | 4-8 hours | 30-60 min | Don't — scope creep |
| Speculative abstraction layer | 2-4 hours | 15-30 min | Don't — YAGNI |

## When to Apply

- Test coverage: edge cases, error paths, boundary conditions
- Input validation: all user-facing and API boundary inputs
- Error handling: descriptive messages, recovery hints, proper propagation
- Documentation: complex logic, non-obvious design decisions
- Migration scripts: forward + rollback, dry-run mode

## When NOT to Apply

- **YAGNI features** — more scope, not more thoroughness
- **Speculative architecture** — abstractions for hypothetical futures
- **Premature optimization** — profile first, optimize second
- **Features not in spec** — completeness = finishing agreed scope, not expanding it
- **A thing that need not exist** — before writing it, climb `code-efficiency-ladder`: does it exist already, does the stdlib or platform cover it, can it be one line

## Decision Framework

```
Is this more work on AGREED scope?
  → YES: Do it. CC effort is cheap. Be thorough.

Is this more SCOPE than agreed?
  → YES: Design-first gate. Don't just do it — discuss first.

Uncertain if in scope?
  → ASK. Clarify before investing effort.
```

The two axes, which is why "prefer thorough" and "stop at the first rung that holds" do not
conflict:

| Axis | Default |
|---|---|
| Quality of what is in scope | Thorough. The marginal cost is near-zero |
| New scope, new abstractions, new dependencies | Lazy. Stop at the first rung that holds |

Where they meet is a simplification that cuts a real corner. Build it lazy, then mark it: the
`// SIMPLIFIED:` convention in `code-quality` records the ceiling and the upgrade trigger, so the
lazy choice stays visible instead of becoming silent debt.

## Cross-References

- **reference:code-efficiency-ladder** — whether the thing should exist at all
- **skill:code-quality** — the `// SIMPLIFIED:` marker for sanctioned shortcuts
