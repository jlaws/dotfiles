---
name: cmd-j-arch
description: "Architecture consultation — API design, distributed patterns, and system design. Use when designing APIs, choosing architecture patterns, or making technology decisions."
disable-model-invocation: true
---

# Architecture Consultation

Before starting, gather diagnostic context:

1. **Detect project architecture** from config files and directory structure — monolith, microservices, serverless, etc.
2. **Identify API patterns** by searching for route definitions, API specs (openapi.yaml, swagger), or GraphQL schemas.
3. **Check architecture documentation** for existing ADRs, design docs, or README architecture sections.
4. **Get scope overview** of the target area (if the user's provided input specifies a component, scope to that; otherwise scan for src/, services/, api/, or similar directories).

For deep architecture guidance, delegate to the `architecture-specialist` agent, passing the diagnostic findings above and the request. It loads its skills (design-first) and the `.agents/references/architecture/` library, then returns specific guidance. Verify its output against the codebase before presenting.

Help with: the user's provided input
