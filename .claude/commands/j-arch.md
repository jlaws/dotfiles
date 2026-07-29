---
name: j-arch
description: "Architecture consultation — API design, distributed patterns, and system design. Use when designing APIs, choosing architecture patterns, or making technology decisions."
argument-hint: "<question-or-task>"
model: opus
effort: xhigh
---

Load skill `analysis-output-patterns` for output structure rules.

Before starting, gather diagnostic context:

1. **Detect project architecture** from config files and directory structure — monolith, microservices, serverless, etc.
2. **Identify API patterns** by searching for route definitions, API specs (openapi.yaml, swagger), or GraphQL schemas.
3. **Check architecture documentation** for existing ADRs, design docs, or README architecture sections.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for src/, services/, api/, or similar directories).

For deep architecture guidance, delegate to the `architecture-specialist` agent via the Task tool, passing the diagnostic findings above and the request. It loads its skills (design-first) and the `references/architecture/` library, then returns specific guidance. Verify its output against the codebase before presenting.

If the consultation reaches a significant architecture decision, offer to capture it as an ADR (see the `architecture-decision-records` reference) in `docs/adr/NNNN-title.md`. Skip minor or easily reversible changes.

Help with: $ARGUMENTS
