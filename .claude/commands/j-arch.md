---
name: j-arch
description: "Architecture consultation — API design, distributed patterns, and system design. Use when designing APIs, choosing architecture patterns, or making technology decisions."
argument-hint: "<question-or-task>"
---

Load skill `analysis-output-patterns` for output structure rules.
Load skill `design-first` for design-before-implementation discipline.

Before starting, gather diagnostic context:

1. **Detect project architecture** from config files and directory structure — monolith, microservices, serverless, etc.
2. **Identify API patterns** by searching for route definitions, API specs (openapi.yaml, swagger), or GraphQL schemas.
3. **Check architecture documentation** for existing ADRs, design docs, or README architecture sections.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for src/, services/, api/, or similar directories).
5. **Check for DECISIONS.md** — review lightweight decision log for recent in-flight choices.

Load relevant references based on the diagnostic context:
- **API design**: `references/architecture/api-design-principles`, `api-design-checklist`, `rest-best-practices`, `rest-api-template`, `graphql-schema-design`, `grpc-examples`, `pagination-patterns` -- contract design, versioning, REST/GraphQL/gRPC conventions
- **Distributed patterns**: `references/architecture/microservices-patterns`, `distributed-communication-patterns`, `message-queue-examples`, `event-sourcing-examples`, `real-time-systems`, `notification-systems` -- service boundaries, async messaging, event flows
- **Resilience & scale**: `references/architecture/caching-strategies`, `retry-patterns`, `background-job-processing`, `saas-multi-tenancy` -- caching, backoff/idempotency, job queues, multi-tenancy
- **Patterns & structure**: `references/architecture/architecture-patterns`, `architecture-decision-records`, `ml-system-design`, `server-examples`, `mcp-server-development`, `deployment` -- pattern selection, ADRs, ML systems, deployment topology
- **Errors & testing**: `references/architecture/error-management`, `error-handling-patterns`, `testing-strategies`, `testing-and-integration` -- error taxonomy, resilience, integration test strategy
- **Process**: `references/workflow/existing-code-discipline` -- reuse-before-adding discipline

Help with: $ARGUMENTS

---

## Decision Logging (Lightweight)

Lightweight, session-level decision log for in-flight technical choices. Complements formal ADRs for smaller decisions.

### When to Log

- Choosing between two or more valid approaches
- Deviating from an established pattern
- "Why didn't you do X?" decisions — anything a future reader would question
- Trade-offs where the alternative was reasonable

### Format

Append to `DECISIONS.md` in the project root (create if absent):

```markdown
## YYYY-MM-DD: [Title]

**Chosen:** [approach]
**Alternatives:** [what else was considered]
**Why:** [reasoning — constraints, trade-offs, context]
**Trade-offs:** [what we gave up]
**Revisit if:** [conditions that would change this decision]
```

### Rules

- **Append-only** — never edit past entries (add a new entry that supersedes)
- **Record contemporaneously** — log when you decide, not after the fact
- **Keep brief** — 3-5 lines per entry; link to longer docs if needed
- **Include context** — decisions without context are useless in 6 months

### Relationship to ADRs

| Scope | Use |
|-------|-----|
| Session/task-level choice | DECISIONS.md (this format) |
| Broad, lasting architectural impact | Formal ADR (`docs/adr/NNNN-title.md`) |

Promote a DECISIONS.md entry to a formal ADR if the decision has broad, lasting impact across the project.
