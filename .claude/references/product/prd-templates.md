# PRD Templates & Patterns

## PRD Structure

| Section | Purpose | Required |
|---------|---------|----------|
| **Problem Statement** | What's broken, for whom, with evidence (metrics, quotes, tickets) | Yes |
| **Proposed Solution** | What we're building and how it solves the problem | Yes |
| **Success Metrics** | Leading indicators (adoption, activation) + lagging (revenue, retention) | Yes |
| **Non-Goals** | Explicitly what this does NOT solve — prevents scope creep | Yes |
| **Acceptance Criteria** | Given/When/Then conditions for "done" | Yes |
| **Launch Plan** | Rollout stages, rollback triggers, monitoring | Yes |
| **Sprint Health Snapshot** | Velocity trend, blockers, scope changes since kickoff | Optional |

## Problem-First Rule

Every PRD starts with the problem + evidence. If you can't fill in all three columns below, you're not ready to write a PRD:

| Column | Example |
|--------|---------|
| **Who** has the problem | "Enterprise customers with >50 team members" |
| **What** they can't do | "Cannot bulk-invite users — must add one at a time" |
| **Evidence** it matters | "37 support tickets/month, 3 churned accounts cited this" |

## Press-Release-Before-PRD

Write a 1-paragraph "press release" as if the feature shipped. If it doesn't sound compelling, rethink the feature:

> **[Product] now lets [user type] do [capability], reducing [pain point] by [expected improvement].** Previously, users had to [old workflow]. Now they can [new workflow] in [time/effort]. Early adopters report [expected outcome].

If you can't write this paragraph convincingly, the feature isn't ready for a PRD.

## Scope Creep Defense

Every proposed addition must pass this filter:

| Question | If No... |
|----------|----------|
| Does it solve the stated problem? | Cut it |
| Do we have evidence users need this? | Cut it |
| Does it have a success metric? | Cut it |
| Can we ship without it? | Defer it |
| Does it increase launch risk? | Defer it |

## Acceptance Criteria Patterns

Use Given/When/Then format for testable criteria:

| Type | Given | When | Then |
|------|-------|------|------|
| **Happy path** | User has admin role | They click "Bulk Invite" and upload CSV | All valid emails receive invitations within 5 min |
| **Edge case** | CSV contains duplicate emails | They upload the file | Duplicates are flagged, unique emails are sent |
| **Error state** | CSV has >500 rows | They upload the file | Error message: "Maximum 500 invitations per batch" |
| **Performance** | 500-row CSV uploaded | Processing begins | All invitations queued within 30 seconds |
| **Rollback** | Feature flag disabled | Any user visits the page | Original single-invite UI is shown |

## PRD Template

```markdown
# PRD: [Feature Name]

**Author:** [name] | **Status:** Draft | **Last Updated:** [date]

## Problem
[Who has this problem? What can't they do? What's the evidence it matters?]

### Evidence
- Support tickets: [count/month]
- User interviews: [N users, key quotes]
- Metrics: [relevant data points]

## Press Release (1 paragraph)
[Write as if the feature shipped — make it compelling or rethink]

## Proposed Solution
[What we're building. Include wireframes/mockups link if available.]

## Success Metrics
| Metric | Current | Target | Timeframe |
|--------|---------|--------|-----------|
| [leading indicator] | [baseline] | [goal] | 30 days |
| [lagging indicator] | [baseline] | [goal] | 90 days |

## Non-Goals
- [What this explicitly does NOT solve]
- [Adjacent problem we're deferring]

## Acceptance Criteria
| Given | When | Then |
|-------|------|------|
| [precondition] | [action] | [expected result] |

## Launch Plan
- **Stage 1:** Internal dogfood (1 week)
- **Stage 2:** Beta — 10% of users (2 weeks)
- **Stage 3:** GA — 100% rollout
- **Rollback trigger:** [error rate >X% OR latency >Yms OR user complaints >Z]

## Open Questions
- [Unresolved decisions]
```
