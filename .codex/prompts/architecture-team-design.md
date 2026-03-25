---
name: architecture-team-design
description: "System design suite — produce architecture documents from multiple specialist perspectives. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /arch instead)."
argument-hint: "<directory-path> <system-description>"
---

Parse arguments: `$ARGUMENTS` must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.

**Do NOT use subagents or parallel agents. Process all design perspectives linearly.**

## Design Process

### Phase 1: Discovery
1. Read the target directory structure and key files
2. Identify existing architecture patterns, APIs, data models
3. Check for existing design docs, ADRs, or README architecture sections

### Phase 2: Multi-Perspective Analysis
Analyze the system from each perspective sequentially:

**2.1 Component Architecture**
- System boundaries and component responsibilities
- Dependencies and interaction patterns
- Data flow between components

**2.2 API Design**
- Public interfaces and contracts
- Versioning strategy
- Error handling patterns

**2.3 Data Architecture**
- Data models and relationships
- Storage strategy and access patterns
- Migration and evolution approach

**2.4 Security & Operations**
- Authentication/authorization model
- Deployment topology
- Observability and failure modes

### Phase 3: Synthesis
Combine findings into a unified architecture document:
- System overview diagram (describe in text/mermaid)
- Component catalog with responsibilities
- Key design decisions and trade-offs
- Risks and open questions

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
