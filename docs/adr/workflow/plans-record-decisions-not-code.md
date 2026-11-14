---
status: accepted
topic: workflow
created: 2026-11-14
updated: 2026-11-14
---
# Plans record decisions, not code

## Context

`writing-plans` required "Complete code" and a junior-engineer bar. Plans grew toward transcripts of the
program. superpowers replaced the same rule in v6.4.2 after models wrote scratch implementations while
planning.

## Decision Drivers

* **Plan time and tokens** spent before any code runs.
* **Executor correctness** on the plan as written.
* **One statement per rule**, so the planning skill does not say the same thing twice.

## Decision

We will have a plan record **the decisions the implementer cannot make alone**: files, interfaces, test
names and assertions, spec values, and commands with expected output. A function body appears only when
the signature, tests, and spec leave the algorithm open. Self-review adds a decision scan and Proportion
(how much of the plan is code).

## Rationale

Upstream measured 9/9 on planted-defect probes for both plan styles, at about a quarter of the time and
a third of the tokens (its measurement on its fixtures, quoted). The completeness principle governs
scope; this rule governs how in-scope work is written down, so the bar is "unambiguous", not "complete".

## Consequences

**Gained**: shorter plans and less plan-time implementation.
**Accepted**: the implementer writes bodies from signatures and tests. SDD's cheap-tier row ("Mechanical, 1-2 files, complete spec") still applies: a signature plus assertions is a complete spec for a mechanical task, and a task whose body is open routes up a tier.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Keep "Complete code" | Plans duplicate the code they describe | 2026-11-14 |
| Compare plan length to spec length | A short spec makes every plan look long | 2026-11-14 |

## Enforcement
- Owner: `.claude/skills/writing-plans/SKILL.md`
- Pinned by: `tests/test_agent_config.py::test_plans_record_decisions_not_code`

## Reversal Conditions
In two consecutive plans, an executor needs a function body the plan did not give to implement a task
correctly, or an SDD task stops after two fix rounds because the plan left the algorithm open.

## Related
- [Plan PRs are sized for review and closed by a diff review](plan-prs-sized-and-closed-by-review.md)
