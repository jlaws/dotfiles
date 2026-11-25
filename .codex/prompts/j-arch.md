---
name: j-arch
description: "Architecture consultation: API design, distributed patterns, and system design. Use when designing APIs, choosing architecture patterns, or making technology decisions."
argument-hint: "<question-or-task>"
---

Before starting, gather diagnostic context:

1. **Detect project architecture** from config files and directory structure: monolith, microservices, serverless, etc.
2. **Identify API patterns** by searching for route definitions, API specs (openapi.yaml, swagger), or GraphQL schemas.
3. **Check architecture documentation** for existing ADRs (glob `docs/adr/**/*.md`), design docs, or README architecture sections.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for src/, services/, api/, or similar directories).

Load the relevant references for the topic, based on the diagnostic context:

Reference library at `~/.agents/references/architecture/`:
- api-design-checklist, api-design-principles, architecture-decision-records
- architecture-patterns, background-job-processing, caching-strategies
- distributed-communication-patterns, error-management
- event-sourcing-examples, graphql-schema-design, grpc-examples
- mcp-client-configuration, mcp-server-development, message-queue-examples
- microservices-patterns, ml-system-design, notification-systems
- pagination-patterns, real-time-systems, rest-api-template
- rest-best-practices, retry-patterns, saas-multi-tenancy
- server-examples, testing-and-integration, testing-strategies

Also see `~/.agents/references/workflow/`:
- existing-code-discipline

For deep architecture guidance, you may instead delegate to the `architecture-specialist` agent, passing the diagnostic findings above and the request. It loads its skills (design-first) and the `~/.agents/references/architecture/` library, then returns specific guidance. Verify its output against the codebase before presenting.

If the consultation reaches a significant architecture decision, offer to capture it as an ADR (see the `architecture-decision-records` reference) in `docs/adr/<topic>/<slug>.md`. If an ADR already covers the decision, update it in place (revise the Decision, add the previous approach to `## Ruled Out` with its reason and date, and bump `updated`) rather than writing a second file. Skip minor or easily reversible changes.

Help with: $ARGUMENTS
