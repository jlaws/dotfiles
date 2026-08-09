# Architecture Decision Records

## Concepts

| Term | Meaning |
|------|---------|
| **Architectural Decision (AD)** | A justified design choice that addresses an architecturally significant requirement. |
| **Architecturally Significant Requirement (ASR)** | A requirement important enough that the decision it drives warrants a record. |
| **Architecture Decision Record (ADR)** | A document capturing a single AD and its rationale — context, options, decision, consequences. |
| **Decision Log** | The full collection of ADRs kept for a project (the `docs/adr/` set). |
| **ADR id** | An ADR's path under `docs/adr/` minus the extension — `data/postgres-primary-store`. There are no ADR numbers. |

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
                     -> Superseded (by <topic>/<slug>)
Proposed -> Rejected
```

Status-header conventions:
- `status: proposed` — under discussion, not yet decided.
- `status: accepted` — the decision is in force.
- `status: rejected` — considered and declined (kept for the record).
- `status: deprecated` — no longer applies, with no direct replacement (add a dated Amendment Log row explaining why).
- `status: superseded` — replaced by a newer decision. The replaced ADR carries `superseded-by: <id>`; the replacing ADR carries `supersedes: [<id>]`. Both link the other from Related Decisions.

## When to Write an ADR

| Write ADR | Skip ADR |
|-----------|----------|
| New framework adoption | Minor version upgrades |
| Database technology choice | Bug fixes |
| API design patterns | Implementation details |
| Security architecture | Routine maintenance |
| Integration patterns | Configuration changes |

## Frontmatter

Every ADR opens with a YAML frontmatter block. It is the machine-readable half of the record — status
and dates are read by globbing, not by parsing prose.

```yaml
---
status: accepted        # proposed | accepted | rejected | deprecated | superseded
topic: data
created: 2026-03-01     # first written; immutable
updated: 2026-03-01     # date of the newest Amendment Log row; equals created when new
deciders: ["@name"]
supersedes: []          # ADR ids, e.g. ["data/mongodb-profiles"]
superseded-by: null     # ADR id, or null
---
```

Two invariants:
- **`created` never changes.** It records when the decision was made, not when the file was last touched.
- **`updated` always equals the newest Amendment Log row's date.** A new ADR with an empty log has `updated` equal to `created`.

## Amendments & Status Transitions

An Accepted ADR's **Decision is immutable**. The record changes only through one of these paths:

| Situation | Action | Result |
|-----------|--------|--------|
| Clarification, corrected detail, or added consequence that does **not** reverse the decision | Add a dated row to the ADR's **Amendment Log** and bump `updated`; leave the Decision text untouched | stays `accepted` |
| The decision itself changes (reversal or material change) | Write a **new ADR** that supersedes the old one; cross-link both | old -> `status: superseded` + `superseded-by: <id>`, new -> `supersedes: [<id>]` |
| Decision no longer relevant, no replacement | Add a dated Amendment Log row explaining why, bump `updated` | `status: deprecated` |

The boundary is simple: **any reversal or material change gets a new ADR** (use the Deprecation ADR template). Minor clarifications get an Amendment Log row. Never rewrite an accepted Decision in place. Superseded and deprecated ADRs stay in the log; they are immutable history, not deletions.

The **Amendment Log** is a fixed section in every ADR (see templates). It starts empty:

```markdown
## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
| YYYY-MM-DD | Clarified retry-budget wording | Ambiguous in review | @name |
```

Every row is paired with bumping `updated` in frontmatter. A row without the bump is an incomplete amendment.

## Templates

Recognized ADR formats include Nygard (the 2011 original), MADR (most widely adopted), the Y-Statement (one-sentence), and ISO/IEC/IEEE 42010. The templates below cover the common ones.

### Standard ADR (MADR Format)

```markdown
---
status: accepted
topic: [topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
deciders: ["@name"]
supersedes: []
superseded-by: null
---
# [Title]

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
- [title](../[topic]/[slug].md)

## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
```

### Lightweight ADR

```markdown
---
status: accepted
topic: [topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
deciders: ["@name"]
supersedes: []
superseded-by: null
---
# [Title]

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

(Single-sentence format — keeps the frontmatter, drops the Amendment Log; amend by superseding.)

### Deprecation ADR

```markdown
---
status: accepted
topic: [topic]
created: YYYY-MM-DD
updated: YYYY-MM-DD
deciders: ["@name"]
supersedes: ["[topic]/[slug]"]
superseded-by: null
---
# Deprecate X in Favor of Y

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

## Naming and Grouping

```
docs/adr/
  README.md            # conventions and this repo's topic list -- not an index
  template.md          # copy this to start an ADR
  data/
    postgres-primary-store.md
    redis-session-cache.md
  api/
    rest-versioning.md
  security/
    oidc-service-auth.md
```

- **Topic** is a single directory level naming the area the decision constrains. Pick names from the repo's own boundaries and record the list in `docs/adr/README.md`. There is no prescribed taxonomy.
- **Slug** is a kebab-case noun phrase naming the decision, unique within its topic. No number, no date, no author.
- **No sequence counter anywhere.** Ordering comes from `created`.
- **Adding an ADR creates exactly one file and edits none.** That is the point of the scheme: a shared counter or a hand-maintained index turns every concurrent ADR write into a merge conflict, because both writers claim the same next number or edit the same index lines.

Two scaffold files sit beside the topic directories:

- `docs/adr/README.md` — the repo's topic list with one line on what each covers, the slug rule, and the amend-vs-supersede rule. It states explicitly that there is no index: discover ADRs with `docs/adr/**/*.md` and read frontmatter.
- `docs/adr/template.md` — the Standard ADR template above, verbatim, ready to copy.

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
- [ ] frontmatter status/created/updated set
- [ ] related ADRs cross-linked both ways
- [ ] Team notified
- [ ] Implementation tickets created

### Definition of Done

A decision is done when it has: **evidence** for the choice, the **criteria and alternatives** considered, **agreement** from the deciders, a written **ADR (documentation)**, and a **realization/review plan** (how it gets built and when it is revisited).

## Do's and Don'ts

- **Write early** - before implementation starts
- **Keep short** - 1-2 pages max
- **Be honest about trade-offs** - include real cons
- **Don't number ADRs** - concurrent writers collide on the next number
- **Don't maintain a hand-edited index** - glob `docs/adr/**/*.md` and read frontmatter
- **Don't change accepted ADRs** - write new ones to supersede
- **Record minor clarifications in the Amendment Log** - not by editing the Decision
- **Don't hide failures** - rejected decisions are valuable
- **Don't be vague** - specific decisions, specific consequences

## Architecture Patterns Reference

Catalog of proven backend architecture patterns (Clean Architecture, Hexagonal Architecture, Domain-Driven Design), when not to reach for each, and their pitfalls. See `architecture-patterns.md`.
