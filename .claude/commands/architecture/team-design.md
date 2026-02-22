---
description: "Multi-agent system design suite — parallel specialist agents produce a complete set of architecture documents."
---

Parse arguments: `$ARGUMENTS` must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.

## Pre-flight

1. **Parse arguments**: extract `directory_path` and `description` from `$ARGUMENTS`.
2. **Create directory** if it doesn't exist (`mkdir -p`).
3. **Check for existing docs**: read any `.md` files already in the directory — this is the current system state for iterative updates.
4. **Detect keywords** in description for conditional skill loading:
   - SaaS / multi-tenant → `architecture:saas-multi-tenancy`
   - search → `data:search-infrastructure`
   - streaming / events → `data:streaming-data-processing`
   - gRPC / protobuf → `architecture:grpc-and-protobuf`
   - real-time → `architecture:real-time-systems`
   - compliance / HIPAA / GDPR / PCI → `security:compliance-and-data-privacy`
   - ML / AI / model → `ai-ml:ml-system-design`
   - GPU → `cloud:gpu-compute-management`
   - accessibility → `frontend:accessibility-testing`
   - frontend / UI → `frontend:design-system-patterns`

## Phase 1: Draft (parallel)

1. **Create team** with TeamCreate (name: `system-design`).
2. **Create 5 tasks** with TaskCreate — each declares file ownership, skills to load, and full context (existing files if iterative + user description). Include the output file structure template (see below) in every task.
3. **Spawn 5 general-purpose agents** and assign tasks via TaskUpdate:

### Agent 1: architecture-analyst
**Owns**: `overview.md`, `architecture.md`, `technology-choices.md`
**Always load skills**: architecture:architecture-patterns, architecture:microservices-patterns, architecture:architecture-decision-records
**Conditional skills**: architecture:saas-multi-tenancy (if SaaS/multi-tenant)
**Focus**: System goals/constraints/stakeholders/NFRs in overview.md. Component architecture with mermaid diagrams and ADRs in architecture.md. Tech stack decisions with comparison tables in technology-choices.md.

### Agent 2: data-and-api-designer
**Owns**: `data.md`, `api.md`
**Always load skills**: architecture:api-design-principles, data:postgresql-table-design, data:nosql-data-modeling, architecture:caching-strategies, architecture:event-sourcing-patterns, architecture:message-queue-patterns
**Conditional skills**: data:search-infrastructure (if search), data:streaming-data-processing (if streaming/events), architecture:grpc-and-protobuf (if gRPC), architecture:real-time-systems (if real-time)
**Focus**: Database selection, schema design, data flow, caching layers in data.md. API surface (REST/GraphQL/gRPC), inter-service contracts, async messaging in api.md.

### Agent 3: infra-planner
**Owns**: `infrastructure.md`, `scalability.md`
**Always load skills**: cloud:serverless-patterns, devops:k8s-manifest-generator, devops:pipeline-design, devops:observability, cloud:cost-optimization, architecture:background-job-processing
**Conditional skills**: ai-ml:ml-system-design (if ML/AI), cloud:gpu-compute-management (if GPU)
**Focus**: Deployment topology, compute strategy, CI/CD, monitoring/SLOs in infrastructure.md. Scaling strategy, capacity planning, performance budget in scalability.md.

### Agent 4: security-reviewer
**Owns**: `security.md`
**Always load skills**: security:security-analysis, security:auth-implementation-patterns, architecture:error-handling-patterns
**Conditional skills**: security:compliance-and-data-privacy (if compliance/HIPAA/GDPR/PCI)
**Focus**: STRIDE threat model, auth/authz strategy, resilience patterns (circuit breaker, retry, graceful degradation), compliance considerations.

### Agent 5: ux-and-features-analyst
**Owns**: `features.md`, `user-experience.md`
**Always load skills**: (none by default)
**Conditional skills**: frontend:accessibility-testing (if accessibility), frontend:design-system-patterns (if frontend/UI)
**Focus**: Feature breakdown with MoSCoW prioritization, use cases in features.md. User journeys, interaction flows, UX considerations in user-experience.md.

**For iterative updates**: Include existing file contents in each agent's task context. Instruct agents to read existing content first, then apply changes while preserving structure and prior decisions. Do NOT rewrite from scratch — update, extend, or revise.

## Phase 2: Cross-Review

After ALL agents complete their drafts (monitor via TaskList):

1. **Send review instructions** to each agent via SendMessage: read ALL other `.md` files in the output directory and flag:
   - Inconsistencies (e.g., data.md says PostgreSQL but infra.md designed for DynamoDB)
   - Missing cross-references (e.g., api.md defines endpoints not covered in features.md)
   - Contradictory assumptions (e.g., security.md assumes JWT but architecture.md says session-based)
2. **Collect conflict reports** from agents via messages.

## Phase 3: Resolve & Finalize

1. **Triage conflicts**: resolve contradictions by making a decision (prefer the specialist's domain — e.g., trust security-reviewer on auth, data-and-api-designer on storage).
2. If a conflict is genuinely ambiguous, ask the user via AskUserQuestion.
3. **Send resolution instructions** to affected agents via SendMessage — instruct them to update their files.
4. Wait for agents to confirm updates.

## Phase 4: Index & Shutdown

1. **Write `README.md`** in the output directory — the lead writes this directly (not an agent). Format:

```markdown
# [System Name] — System Design

> [One-line description from user input]

Last updated: [date]

## Design Documents

| Document | Description |
|----------|-------------|
| [overview.md](./overview.md) | Goals, constraints, stakeholders, NFRs |
| [architecture.md](./architecture.md) | Component architecture, patterns, ADRs |
| [technology-choices.md](./technology-choices.md) | Tech stack decisions with rationale |
| [data.md](./data.md) | Data model, storage, caching, data flow |
| [api.md](./api.md) | API surface, contracts, protocols |
| [infrastructure.md](./infrastructure.md) | Deployment, compute, CI/CD, monitoring |
| [scalability.md](./scalability.md) | Scaling strategy, capacity, performance |
| [security.md](./security.md) | Threat model, auth, resilience, compliance |
| [features.md](./features.md) | Feature breakdown, use cases, priorities |
| [user-experience.md](./user-experience.md) | User journeys, interaction flows, UX |
```

2. **Shut down all agents** via SendMessage (type: shutdown_request).
3. **Delete team** via TeamDelete.
4. **Present summary** to user: list all files created/updated with brief descriptions.

## Output File Template

Every generated `.md` file MUST follow this structure:

```markdown
# [System Name]: [Aspect Title]

## Overview
Brief summary of this aspect of the system.

## [Domain-specific sections]
(varies per file — architecture diagrams, data models, API specs, etc.)

## Key Decisions
| Decision | Choice | Rationale | Alternatives Considered |
|----------|--------|-----------|------------------------|
| ... | ... | ... | ... |

## Open Questions
- [ ] Question 1
- [ ] Question 2

## References
- Links to related design docs in this suite
```

$ARGUMENTS
