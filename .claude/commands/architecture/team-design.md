---
name: team-design
description: "Multi-agent system design suite — parallel specialist agents produce architecture documents. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /arch instead)."
argument-hint: "<directory-path> <system-description>"
---

Parse arguments: `$ARGUMENTS` must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.

## Parallel System Design

### Step 1: Gather Context

1. Scan the target directory for existing architecture (config files, directory structure, README)
2. Identify key components, boundaries, and integration points
3. Determine system type (monolith, microservices, serverless, etc.)

### Step 2: Dispatch Parallel Subagents

Dispatch parallel subagents to cover different architecture perspectives:

1. **component-analyst** (Explore) — Map components, dependencies, and module boundaries. Produce component diagram description.
2. **api-analyst** (Explore) — Identify APIs, data flows, and integration contracts. Document endpoint inventory and data models.
3. **infra-analyst** (Explore) — Analyze deployment, infrastructure patterns, and operational concerns. Document infrastructure topology.
4. **quality-analyst** (Explore) — Assess testing strategy, security boundaries, and performance characteristics. Identify architectural risks.

Each subagent receives the directory path, system description, and any existing architecture docs as context.

### Step 3: Synthesize Architecture Documents

After all subagents return, synthesize findings into:

```markdown
## System Architecture — {system-description}

### Overview
[High-level system description and architectural style]

### Component Map
[Components, boundaries, and dependencies from component-analyst]

### API & Data Flow
[Endpoints, contracts, and data models from api-analyst]

### Infrastructure
[Deployment topology and operational concerns from infra-analyst]

### Quality & Risk Assessment
[Testing strategy, security boundaries, performance from quality-analyst]

### Architectural Decisions
[Key trade-offs and recommendations]

### Open Questions
[Unresolved design decisions needing stakeholder input]
```
