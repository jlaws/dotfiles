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
