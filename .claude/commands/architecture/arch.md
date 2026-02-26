---
name: arch
description: "Architecture consultation — API design, distributed patterns, and system design. Use when designing APIs, choosing architecture patterns, or making technology decisions."
argument-hint: "<question-or-task>"
---

Before invoking the subagent, gather diagnostic context:

1. **Detect project architecture** from config files and directory structure — monolith, microservices, serverless, etc.
2. **Identify API patterns** by searching for route definitions, API specs (openapi.yaml, swagger), or GraphQL schemas.
3. **Check architecture documentation** for existing ADRs, design docs, or README architecture sections.
4. **Get scope overview** of the target area (if $ARGUMENTS specifies a component, scope to that; otherwise scan for src/, services/, api/, or similar directories).

Use the architecture-specialist subagent to help with: $ARGUMENTS
