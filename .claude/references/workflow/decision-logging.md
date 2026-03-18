
# Decision Logging (Lightweight)

Lightweight, session-level decision log for in-flight technical choices. Complements formal ADRs (`references/architecture/architecture-decision-records.md`) for smaller decisions.

## When to Log

- Choosing between two or more valid approaches
- Deviating from an established pattern
- "Why didn't you do X?" decisions — anything a future reader would question
- Trade-offs where the alternative was reasonable

## Format

Append to `DECISIONS.md` in the project root (create if absent):

```markdown
## YYYY-MM-DD: [Title]

**Chosen:** [approach]
**Alternatives:** [what else was considered]
**Why:** [reasoning — constraints, trade-offs, context]
**Trade-offs:** [what we gave up]
**Revisit if:** [conditions that would change this decision]
```

## Rules

- **Append-only** — never edit past entries (add a new entry that supersedes)
- **Record contemporaneously** — log when you decide, not after the fact
- **Keep brief** — 3-5 lines per entry; link to longer docs if needed
- **Include context** — decisions without context are useless in 6 months

## Relationship to ADRs

| Scope | Use |
|-------|-----|
| Session/task-level choice | DECISIONS.md (this format) |
| Broad, lasting architectural impact | Formal ADR (`docs/adr/NNNN-title.md`) |

Promote a DECISIONS.md entry to a formal ADR if the decision has broad, lasting impact across the project.
