---
name: j-system-design
description: "Comprehensive system design suite — produce architecture documents covering multiple perspectives. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /j-arch instead)."
argument-hint: "<system-description>"
---

Load skill `design-first` for design-before-implementation discipline.
Load skill `analysis-output-patterns` for output structure rules.

Parse arguments: `$ARGUMENTS` is a freeform system description that may contain file or directory references inline.
- The entire string is the system description -- do not strip paths from it.
- Scan the text for path-like tokens (tokens containing `/`, or matching extensions like `.md`, `.ts`, `.py`, `.go`, `.rs`, `.json`, `.yaml`, `.toml`). These are explicit targets to include in document discovery.
- If `$ARGUMENTS` is empty, ask the user for a system description.

**You may delegate independent design perspectives to specialist agents (`architecture-specialist`, `security-reviewer`, `data-engineer`) via the Task tool and run them in parallel. Synthesize their findings and verify each against the codebase before presenting.**

## Design Process

### Phase 1: Discovery

**Step 1 -- Scan for candidate documentation:**
1. Glob for common doc directories: `docs/`, `doc/`, `documents/`, `documentation/`, `design/`, `adr/`, `adrs/`, `architecture/`
2. Glob for root markdown files: `*.md` in project root (README, CONTRIBUTING, ARCHITECTURE, etc.)
3. Glob for design doc patterns: `**/ADR-*.md`, `**/design-*.md`, `**/RFC-*.md`
4. Include any file or directory paths found inline in `$ARGUMENTS`

**Step 2 -- Filter by relevance to the system description:**
1. First pass: filter candidates by file/directory names and paths -- keep files whose names relate to the described system
2. Second pass: for ambiguous candidates, use Grep to search file contents for keywords from the system description
3. Discard files that are clearly unrelated (e.g. skip `docs/billing.md` when designing an auth system)

**Step 3 -- Read selected files and explore the codebase:**
1. Read the relevant documentation files selected above
2. Read the project structure and key source files
3. Identify existing architecture patterns, APIs, data models

**Step 4 -- Delegate perspectives to specialist agents (recommended when the system is large):**
Hand each Phase 2 perspective to its agent via the Task tool, in parallel — they load the relevant skills + `references/` libraries and return findings:
- Component Architecture, API Design -> `architecture-specialist`
- Data Architecture -> `data-engineer`
- Security & Operations -> `security-reviewer` (security) + `devops-engineer` (deployment/observability)

Synthesize their findings in Phase 4 and verify each against the codebase. For small systems, analyze the perspectives inline instead.

### Phase 2: Multi-Perspective Analysis
Analyze the system from each perspective (inline, or delegated in parallel per Step 4):

**2.1 Component Architecture**
- System boundaries and component responsibilities
- Dependencies and interaction patterns (who calls whom, sync vs async)
- Data flow between components
- Identify god classes, circular dependencies, unclear ownership
- Map the dependency graph — are layers clean or tangled?

**2.2 API Design**
- Public interfaces and contracts
- Versioning strategy and backwards compatibility approach
- Error handling patterns and error taxonomy
- Input validation and sanitization boundaries
- Pagination, rate limiting, idempotency for external APIs
- Are APIs designed for the consumer or the implementation?

**2.3 Data Architecture**
- Data models and relationships (ER diagram or description)
- Storage strategy and access patterns (read-heavy? write-heavy? mixed?)
- Migration and evolution approach
- Data consistency model (strong, eventual, causal)
- Data lifecycle — creation, transformation, archival, deletion
- Sensitive data identification and handling

**2.4 Security & Operations**
- Authentication/authorization model and trust boundaries
- Deployment topology (single region? multi? edge?)
- Observability: metrics, logging, tracing, alerting
- Failure modes and recovery strategy
- Capacity planning and scaling approach
- Incident response — what can be rolled back, what can't

### Phase 3: Cross-Cutting Analysis

Analyze concerns that span multiple perspectives:

**3.1 Domain Pipeline Analysis**
For cross-cutting issues, trace the full path:
- **Performance**: Request path → DB queries → response. Where are the bottlenecks?
- **Security**: User input → validation → processing → storage. Where are the trust boundaries?
- **Error propagation**: Failure point → error handling → user feedback. Are errors meaningful?

**3.2 Integration Points**
- External service dependencies and failure modes
- Message queues, event buses, or async communication patterns
- Shared state and coordination mechanisms

### Phase 4: Synthesis

Combine findings into a unified architecture document:
- System overview diagram (describe in text/mermaid)
- Component catalog with responsibilities
- Key design decisions and trade-offs
- Risks and open questions
- Implementation priority order (what to build first and why)

### Phase 5: Implementation Planning

Produce an actionable implementation plan:
1. **Task breakdown** — ordered list of implementation tasks with dependencies
2. **Risk-first ordering** — tackle the hardest/riskiest parts first
3. **Vertical slices** — each task should produce something testable end-to-end
4. **Test strategy** — what tests are needed at each level (unit, integration, e2e)

---

## Design Review Checklist

After completing the multi-perspective analysis, review your design against these criteria:

### Spec Compliance
- Does the design address ALL stated requirements?
- Are there requirements the design missed or misinterpreted?
- Is there unnecessary complexity beyond what was asked?

### Quality Assessment
- **Simplicity**: Is this the simplest design that meets requirements?
- **Separation of concerns**: Are responsibilities clearly divided?
- **Extensibility**: Can the design accommodate likely future changes?
- **Consistency**: Does it align with existing patterns in the codebase?

### Risk Assessment
- What are the failure modes? How does the system recover?
- What are the scalability bottlenecks?
- What are the security boundaries and trust assumptions?
- What are the hardest parts to implement? To test?

### Trade-offs Documentation
For each significant design decision:
- **Chosen approach** and why
- **Alternatives considered** and why rejected
- **Trade-offs accepted**
- **Conditions that would change this decision**

### Red Flags — Redesign Before Implementing
- Component has multiple unrelated responsibilities
- Circular dependencies between components
- Design requires coordination between components for basic operations
- No clear error handling strategy
- Design optimizes for hypothetical future requirements over current needs
- Can't explain a component's purpose in one sentence
- "It depends" is the answer to most questions about the design

---

## Self-Review Before Presenting

Review your work with fresh eyes before presenting to the user:

**Completeness:**
- Did I analyze every perspective thoroughly?
- Did I miss any requirements or constraints?
- Are there edge cases or failure modes I didn't consider?

**Quality:**
- Is the design document clear and actionable?
- Are diagrams/descriptions precise enough to implement from?
- Did I follow existing codebase patterns where appropriate?

**Discipline:**
- Did I avoid overbuilding (YAGNI)?
- Is the design the simplest thing that could work?
- Did I only design what was requested?

Fix any issues found during self-review before presenting.

## Report Format

When done, present:
- What you analyzed (scope, files read, patterns identified)
- The architecture document (Phase 4 synthesis)
- Key design decisions and why
- Implementation plan (Phase 5)
- Open questions requiring user input
- Risks and concerns
