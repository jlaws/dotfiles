# Architecture Decision Records

## Concepts

| Term | Meaning |
|------|---------|
| **Architectural Decision (AD)** | A justified design choice that addresses an architecturally significant requirement. |
| **Architecturally Significant Requirement (ASR)** | A requirement important enough that the decision it drives warrants a record. |
| **Architecture Decision Record (ADR)** | A document capturing a single AD and its rationale — context, options, decision, consequences. |
| **Decision Log** | The full collection of ADRs kept for a project (the `docs/adr/` set). |

### Architectural Significance Test

Write an ADR when a decision meets any of these — it is an ASR:
- Affects the system's structure or a component boundary.
- Cross-cutting: touches multiple components or teams.
- Hard or costly to reverse later.
- Involves a non-obvious trade-off worth explaining to future readers.

If none apply, skip the ADR (see the table below).

## ADR Lifecycle

```
Proposed -> Accepted -> Deprecated
                     -> Superseded (by ADR-NNNN)
Proposed -> Rejected
```

Status-header conventions:
- `Status: Proposed` — under discussion, not yet decided.
- `Status: Accepted` — the decision is in force.
- `Status: Rejected` — considered and declined (kept for the record).
- `Status: Deprecated` — no longer applies, with no direct replacement (add a dated note explaining why).
- `Status: Superseded by ADR-NNNN` — replaced by a newer decision. The replacing ADR carries `Supersedes ADR-NNNN` in its Status and Related Decisions.

## When to Write an ADR

| Write ADR | Skip ADR |
|-----------|----------|
| New framework adoption | Minor version upgrades |
| Database technology choice | Bug fixes |
| API design patterns | Implementation details |
| Security architecture | Routine maintenance |
| Integration patterns | Configuration changes |

## Amendments & Status Transitions

An Accepted ADR's **Decision is immutable**. The record changes only through one of these paths:

| Situation | Action | Status |
|-----------|--------|--------|
| Clarification, corrected detail, or added consequence that does **not** reverse the decision | Add a dated row to the ADR's **Amendment Log**; leave the Decision text untouched | stays `Accepted` |
| The decision itself changes (reversal or material change) | Write a **new ADR** that supersedes the old one; cross-link both | old -> `Superseded by ADR-NNNN`, new -> `Supersedes ADR-NNNN` |
| Decision no longer relevant, no replacement | Mark deprecated with a dated note | `Deprecated` |

The boundary is simple: **any reversal or material change gets a new ADR** (use the Deprecation ADR template). Minor clarifications get an Amendment Log row. Never rewrite an accepted Decision in place. Superseded and deprecated ADRs stay in the log; they are immutable history, not deletions.

The **Amendment Log** is a fixed section in every ADR (see templates). It starts empty:

```markdown
## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
| YYYY-MM-DD | Clarified retry-budget wording | Ambiguous in review | @name |
```

## Templates

Recognized ADR formats include Nygard (the 2011 original), MADR (most widely adopted), the Y-Statement (one-sentence), and ISO/IEC/IEEE 42010. The templates below cover the common ones.

### Standard ADR (MADR Format)

```markdown
# ADR-NNNN: [Title]

## Status
Accepted

## Context
[Why we needed to decide. Include constraints, requirements, team experience.]

## Decision Drivers
* **Must have X** for Y reason
* **Should support Z** to reduce complexity

## Considered Options

### Option 1: [Name]
- **Pros**: ...
- **Cons**: ...

### Option 2: [Name]
- **Pros**: ...
- **Cons**: ...

## Decision
We will use **[choice]**.

## Rationale
[Why this option best fits the decision drivers.]

## Consequences

### Positive
- [benefit]

### Negative
- [cost/risk]

## Implementation Notes
- [specific guidance]

## Related Decisions
- ADR-NNNN: [title]

## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
```

### Lightweight ADR

```markdown
# ADR-NNNN: [Title]

**Status**: Accepted | **Date**: YYYY-MM-DD | **Deciders**: @names

## Context
[1-2 paragraphs on the problem]

## Decision
[What we decided]

## Consequences
**Good**: [benefits]
**Bad**: [costs]
**Mitigations**: [how to address the bad]

## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
```

### Y-Statement Format

```markdown
In the context of **[situation]**,
facing **[problem]**,
we decided for **[choice]**
and against **[alternatives]**,
to achieve **[goals]**,
accepting that **[tradeoff]**.
```

(Single-sentence format — no Amendment Log; amend by superseding.)

### Deprecation ADR

```markdown
# ADR-NNNN: Deprecate X in Favor of Y

## Status
Accepted (Supersedes ADR-NNNN)

## Context
[Why the original decision no longer serves us]

## Migration Plan
1. Phase 1 (Week 1-2): Dual-write
2. Phase 2 (Week 3-4): Backfill + validate
3. Phase 3 (Week 5): Switch reads
4. Phase 4 (Week 6): Remove old writes, decommission

## Lessons Learned
- [What we'd do differently]

## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
```

## Directory Structure

```
docs/adr/
  README.md              # Index and guidelines
  template.md            # Team's ADR template
  0001-use-postgresql.md
  0002-caching-strategy.md
  0003-mongodb-profiles.md  # [DEPRECATED]
  0020-deprecate-mongodb.md # Supersedes 0003
```

## Review Checklist

### Before Submission
- [ ] Context clearly explains the problem
- [ ] All viable options considered
- [ ] Pros/cons balanced and honest
- [ ] Consequences (positive and negative) documented

### During Review
- [ ] At least 2 senior engineers reviewed
- [ ] Affected teams consulted
- [ ] Security and cost implications documented
- [ ] Reversibility assessed

### After Acceptance
- [ ] ADR index updated
- [ ] Team notified
- [ ] Implementation tickets created

### Definition of Done

A decision is done when it has: **evidence** for the choice, the **criteria and alternatives** considered, **agreement** from the deciders, a written **ADR (documentation)**, and a **realization/review plan** (how it gets built and when it is revisited).

## Do's and Don'ts

- **Write early** - before implementation starts
- **Keep short** - 1-2 pages max
- **Be honest about trade-offs** - include real cons
- **Don't change accepted ADRs** - write new ones to supersede
- **Record minor clarifications in the Amendment Log** - not by editing the Decision
- **Don't hide failures** - rejected decisions are valuable
- **Don't be vague** - specific decisions, specific consequences

## Architecture Patterns Reference

Catalog of proven backend architecture patterns (Clean Architecture, Hexagonal Architecture, Domain-Driven Design), when not to reach for each, and their pitfalls. See `architecture-patterns.md`.
