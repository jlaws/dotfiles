---
status: accepted
topic: workflow
created: 2026-09-12
updated: 2026-09-12
deciders: ["@jlaws"]
supersedes: []
superseded-by: null
---
# Thorough about quality dimensions, lazy about new scope

## Context

`references/workflow/completeness-principle.md` opens: "When Claude Code effort is low relative to
human effort, prefer thorough over 'good enough.' The marginal cost of completeness is near-zero."

Adopting ponytail's seven-rung code efficiency ladder introduces a rule that reads as the opposite:
"Stop at the first rung that holds." Landing the ladder without reconciling the two would leave two
references in the same directory contradicting each other, and a reader with no way to tell which
applies.

The repo had no before-you-write stopping rule at all. `ladder` appeared only as RTK's search/tool
ladder; `one implementation` and `reinvent` were absent; YAGNI appeared in seven files but always as
a *scope* rule, never as a reuse-before-write reflex. `code-quality/SKILL.md` carried a single
adjacent line: "Premature abstraction -- wait for 2+ concrete implementations."

The ladder also carries a measured risk. ponytail publishes a correctness regression from its own
rules: email-validation accuracy 100% to 79% on gpt-4.1, traced to the model reaching for the
laziest stdlib helper (`parseaddr` accepts `"@missing-local.com"` because it does not require a
local part). Its own diagnosis: "a genuine (if minor) cost of pushing toward one-liners."

## Decision Drivers

* **Must not weaken safety-relevant code** — validation, error handling, security, accessibility.
* **Must resolve the apparent contradiction in writing**, not leave it to the reader.
* **Should keep fidelity with the upstream source**, so the ledger entry can describe a faithful
  adoption rather than a fork.

## Considered Options

### Option 1: Ladder-first everywhere
- **Pros**: One rule, no conditional.
- **Cons**: Contradicts the completeness principle outright, and the measured regression lands
  squarely on validation code.

### Option 2: Exclude parsing and validation paths from the ladder
- **Pros**: Strongest guarantee against the measured failure class.
- **Cons**: Rungs 3-5 (stdlib, native, installed dep) are most useful precisely on parsing. Blunts
  the win where it is largest, and forks the upstream ruleset.

### Option 3: Drop rung 6 ("Can it be one line?")
- **Pros**: Removes the rung ponytail's own defect analysis names.
- **Cons**: Breaks rung-for-rung fidelity, so the adoption can no longer be described as faithful —
  and the defect is really about *which* helper, not about line count.

### Option 4: Adopt in full with ponytail's own guards, and write the axis boundary
- **Pros**: Faithful adoption. Uses the mitigations the upstream added in response to its own
  measurement, which it then measured at 100% safe (20/20) on Claude models.
- **Cons**: Relies on a guard holding rather than on a structural exclusion.

## Decision

We will use **Option 4**.

The boundary: **thorough about the quality dimensions of work already in scope; lazy about new
scope, new abstractions, and new dependencies.** Recorded in `completeness-principle.md` under
"Where Thoroughness Stops", with the ladder as the other half.

Two guards ship with the ladder, both ponytail's own:
1. "Two options the same size? Take the one that is correct on edge cases. Lazy means writing less
   code, not picking the flimsier algorithm."
2. The never-simplify-away list: input validation at trust boundaries, error handling that prevents
   data loss, security, accessibility, anything explicitly requested.

## Rationale

The contradiction is apparent, not real — the two rules govern different axes, and naming the axes
dissolves it. On the risk: ponytail's regression is on non-Claude models, and on Claude its same
audit shows improvement (haiku 35/40 to 40/40, sonnet 0/40 to 40/40, opus 39/40 to 40/40). This repo
targets Claude. Option 2's exclusion would also cost the most-measured part of the win to guard
against a failure this repo's target models did not exhibit.

The guards are not decoration, and there is evidence for that: ponytail's benchmark included a
paraphrase arm that dropped the carve-outs, and it scored 95% safe (19/20) against the full
ruleset's 100% (20/20). The carve-outs are the safety margin, which is why a test pins them.

## Consequences

### Positive
- Rungs 2-5 gain an owner; reuse-before-write becomes a stated reflex rather than an implicit habit.
- Faithful to upstream, so the ledger can describe provenance precisely, including the #217 dating
  of rung 2.

### Negative
- The safety carve-outs are now load-bearing text. If they are edited away the ladder becomes
  unsafe, which is why `tests/test_agent_config.py` pins them.
- The measured win is third-party (n=4, one model, one repo) and is on the code axis only — 44% of
  baseline on coding prompts against 87% on explanation-only prompts.
- The ruleset costs more than it saves on short interactions and on reasoning models. Stated in the
  reference rather than left for a reader to discover.

## Implementation Notes
- Owner: `references/workflow/code-efficiency-ladder.md`.
- Consumers point at it: `skills/code-quality`, `agents/code-reviewer`, `commands/j-diff-review`.
  `tests/test_reference_tree.py` enforces reachability, but transitively — a cross-reference from
  `context-efficiency.md` alone satisfies it. The consumer pointers are therefore a discoverability
  requirement, not a test requirement: a reference reachable only from another reference is indexed
  but never reached from where the decision is actually made. `LADDER_OWNERS` in
  `tests/test_agent_config.py` pins them instead.
- Sanctioned shortcuts use the `// SIMPLIFIED:` marker, owned by `code-quality`, carrying both a
  ceiling and an upgrade trigger.

## Reversal Conditions

Reverse if the edge-case-correctness guard proves insufficient in practice — specifically, if a lazy
stdlib pick ships an edge-case hole in this repo's own code. The narrower Option 2 is the fallback,
not a full revert.

## Related Decisions
- [Structure preference is conditional](../context-efficiency/structure-is-conditional.md)

## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
