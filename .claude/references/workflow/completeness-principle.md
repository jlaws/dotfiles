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

## Where Thoroughness Stops

Thoroughness applies to the **quality dimensions** of work already in scope — test coverage, input
validation, error context, docs, rollback. It does not license new scope. The two "Don't" rows above
are the boundary, and they generalize:

| Axis | Default |
|---|---|
| Quality of what is in scope | Thorough. The marginal cost is near-zero |
| New scope, new abstractions, new dependencies | Lazy. Stop at the first rung that holds |

So "prefer thorough over good enough" and "stop at the first rung that holds" are not in tension —
they govern different axes. Reach for `code-efficiency-ladder` to decide whether a thing should
exist; reach for this page to decide how well to build it once it should.

The one place they genuinely meet is a simplification that cuts a real corner. Build it lazy, then
mark it: the `// SIMPLIFIED:` convention in `code-quality` records the ceiling and the upgrade
trigger, so the lazy choice stays visible instead of becoming silent debt.

## Cross-References

- **reference:code-efficiency-ladder** — whether the thing should exist at all
- **skill:code-quality** — the `// SIMPLIFIED:` marker for sanctioned shortcuts
