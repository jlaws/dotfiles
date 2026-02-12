---
name: architecture-decision-records
description: Write and maintain Architecture Decision Records (ADRs) following best practices for technical decision documentation. Use when documenting significant technical decisions, reviewing past architectural choices, or establishing decision processes.
---

# Architecture Decision Records

## ADR Lifecycle

```
Proposed -> Accepted -> Deprecated -> Superseded
              |
           Rejected
```

## When to Write an ADR

| Write ADR | Skip ADR |
|-----------|----------|
| New framework adoption | Minor version upgrades |
| Database technology choice | Bug fixes |
| API design patterns | Implementation details |
| Security architecture | Routine maintenance |
| Integration patterns | Configuration changes |

## Templates

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

## adr-tools

```bash
brew install adr-tools
adr init docs/adr
adr new "Use PostgreSQL as Primary Database"
adr new -s 3 "Deprecate MongoDB in Favor of PostgreSQL"
adr generate toc > docs/adr/README.md
adr link 2 "Complements" 1 "Is complemented by"
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

## Do's and Don'ts

- **Write early** - before implementation starts
- **Keep short** - 1-2 pages max
- **Be honest about trade-offs** - include real cons
- **Don't change accepted ADRs** - write new ones to supersede
- **Don't hide failures** - rejected decisions are valuable
- **Don't be vague** - specific decisions, specific consequences
