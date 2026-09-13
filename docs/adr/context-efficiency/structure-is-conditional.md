---
status: accepted
topic: context-efficiency
created: 2026-09-12
updated: 2026-09-13
deciders: ["@jlaws"]
---
# Structure preference is conditional on what the question asks for

## Context

The always-loaded configs and `references/workflow/context-efficiency.md` both instructed an
unconditional preference for structured output. `.claude/CLAUDE.md` read "Prefer bullets, tables,
and code over prose"; the reference's `Token Density by Format` section closed with "If information
can be a table, make it a table."

That rule is re-sent on every request in every repo, and two independent measurements say it is
wrong for one common prompt class:

- Chisle's only backfire across 20 live tasks was a "summarize/compare REST vs GraphQL" prompt
  answered with headed pro/con walls and "Pick REST if / Pick GraphQL if" scaffolding. Measured at
  173% of a no-tool baseline; after rewriting the rule, 93% (n=3 per arm).
- claude-token-efficient, from the opposite direction: "Words drop more consistently than tokens.
  Markdown and structure tokens partly offset word savings, so word-count benchmarks overstate the
  token (cost) win."

Both are third-party numbers on third-party suites, at small n. What makes them actionable is not
the magnitude but the mechanism, which is checkable by inspection: headed scaffolding creates
sections that then have to be filled, so the format decides the content.

## Decision Drivers

* **Must not degrade answers** — this rule is loaded in every session; a wrong change is expensive
  and diffuse.
* **Must keep the density ordering**, which is correct and independently useful for information
  already decided to be included.
* **Should be statable in one line** in an always-loaded file.

## Decision

We will make the structure preference **conditional**. The density ordering stays and applies to
information already decided to be included. A second rule states that structure the question did not
ask for is a net cost even in table form, with "answer at the question's altitude" as the operative
instruction. The always-loaded configs point at `context-efficiency.md` rather than restating either
half.

## Rationale

The two claims are not in conflict, which is why the absolute form looked right for so long. Density
is per unit of information *carried*; the failure mode is information *manufactured*. Stating only
the first half reads as a licence for the second. A conditional rule is also the only option that
survives the "what would make me reverse this" test with a concrete answer.

## Consequences

**Gained**: The measured failure mode is addressed at its cause rather than by removing a useful
rule. One owner for the rule (`context-efficiency.md`), pinned by a test, instead of three
restatements drifting apart.

**Accepted**: Requires judgment at answer time — a model that misjudges altitude now has a rule that
under-specifies rather than one that over-specifies. The supporting numbers are a competitor's,
measured on its own suite: n=1 per cell on the 20-task ledger, n=3 per arm on the re-measure quoted
above. We adopt the mechanism, not the magnitude, and the ledger entries say so.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Keep the unconditional table preference | Leaves the measured failure mode in place, in a rule that fires constantly. | 2026-09-12 |
| Drop the structure preference entirely | Throws away a real win. Tables *are* denser for comparisons and decision matrices, which is most of what this repo's references are made of. | 2026-09-12 |

## Enforcement
- Owner: `references/workflow/context-efficiency.md`, `Token Density by Format`.
- `.claude/CLAUDE.md`, `.codex/AGENTS.md`, and `.gemini/GEMINI.md` carry a pointer, not a copy;
  `tests/test_agent_config.py` pins that.

## Reversal Conditions

Reverse if a measurement on this repo's own transcripts shows headed structure is not costing
tokens here — the corpus access for that already exists in
`.claude/skills/skill-audit/scripts/adoption.py`.

## Related
- [Code ladder versus the completeness principle](../workflow/code-ladder-vs-completeness.md)
- [ADRs are living documents](../workflow/adrs-are-living-documents.md)
