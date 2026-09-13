---
status: accepted
topic: workflow
created: 2026-09-12
updated: 2026-09-12
deciders: ["@jlaws"]
supersedes: []
superseded-by: null
---
# Reference trees share a section set, not a body

## Context

`.claude/references/` and `.agents/references/` hold the same files at the same relative paths.
`.claude/` serves Claude Code; `.agents/` is what Codex and Gemini read.

`3547c54` (#76) rewrote the Claude tree for the Claude 5 generation under a stated keep test — "a
section stays if a strong model asked cold would answer differently or less specifically" — and
ported only correctness fixes back, leaving `.agents/` deliberately longer. Nothing pinned the
result. By the time this decision was taken, 61 of 192 pairs had diverged and 39 differed
*structurally*: a whole section present in one tree and absent from the other.

Nothing catches it. `tests/test_agent_config.py` pins skill, command, and agent *name sets* but has
never covered references. `audit.py` is Claude-tree-only by its own docstring, so it validates
reachability and link resolution inside `.claude/` and never looks at `.agents/`.

The cost is asymmetric and invisible: a Codex user searching for a topic the Claude tree documents
finds nothing, with no error anywhere to tell either of us.

## Decision Drivers

* **Must catch a section that exists in one tree only** — that is the entire observed failure mode.
* **Must allow per-harness adaptation inside a section** — the trees serve different tools and
  different model generations, and forcing identical prose would be a regression.
* **Should need no maintenance** — a policy with a growing exception list stops being a policy.

## Considered Options

### Option 1: Byte-identical trees
- **Pros**: trivially checkable; zero ambiguity about what "in sync" means.
- **Cons**: destroys legitimate adaptation. `workflow/context-efficiency.md` says `` `Explore` `` in
  the Claude tree and "a dispatched search subagent" in the shared tree, because Codex and Gemini have
  no tool by that name. Byte-identity forces one of those two readers to get a wrong tool name. The
  same problem applies to model IDs and harness-specific paths.

### Option 2: Heading-set parity
- **Pros**: catches all 39 observed structural divergences; permits prose, tool names, model IDs, and
  code-versus-summary differences inside a section; cheap to express as one loop.
- **Cons**: two trees can state opposite things under the same heading and still pass. This check
  constrains structure, not correctness.

### Option 3: File-presence parity only
- **Pros**: cheapest possible check.
- **Cons**: catches none of the 39. Every one of those files exists in both trees already.

## Decision

We will use **heading-set parity**. For every path under `.claude/references/`, the `.agents/`
file at the same path exists and yields the same ordered list of `##`/`###`/`####` headings.
Bodies may differ.

The check ships without an exception list.

## Rationale

Headings are the table of contents a reader navigates by. A topic documented in one tree and missing
from the other is the failure we actually observed 39 times; a topic worded differently in each tree
is the adaptation we want to keep. Heading-set parity is the line between those two, and it is the
only one of the three options that falls on the right side of both.

Shipping without an exception list is deliberate. An exception list turns "the trees agree" into "the
trees agree except where someone once found it inconvenient," and the drift this decision fixes grew
precisely because nothing forced the question.

## Consequences

### Positive
- The 39 structural divergences get fixed once and cannot silently return.
- A Claude-only reference edit now fails the suite until it is mirrored, which surfaces the decision
  at edit time rather than months later.
- `.agents/` gains correctness fixes that had been landing only in `.claude/`.

### Negative
- Every future reference change costs an edit in both trees. That is the point, but it is a real tax.
- Structural divergence has no escape hatch. A genuine need for one will require amending this
  decision rather than adding an entry somewhere.
- The check says nothing about whether the two bodies under a shared heading agree. Two trees can
  contradict each other and pass.

## Implementation Notes

- `test_reference_trees_expose_the_same_sections` in `tests/test_agent_config.py` is the enforcement
  point. It compares ordered heading lists, so a reordering fails too. Headings inside fenced code
  blocks are excluded: a fenced block can hold a markdown *example* — the ADR template, a sample
  CHANGELOG — and that is body content this decision leaves free.
- `test_reference_trees_hold_the_same_files` pins the file set in both directions. The section check
  walks the Claude tree, so on its own it would let an `.agents`-only reference through, and nothing
  else in the repo reads that tree.
- Section parity does not extend to reference bodies pasted elsewhere, and three references are:

  | Reference | Pasted into | Pinned by |
  |---|---|---|
  | `workflow/existing-code-discipline.md` | `.codex/prompts/j-diff-review.md`, `.agents/skills/cmd-j-diff-review/SKILL.md` | `test_existing_code_discipline_is_one_document_in_every_tree`, section set **and** body |
  | `security/security-analysis.md` | `.agents/skills/cmd-j-audit/SKILL.md`, `.codex/prompts/j-audit.md` | nothing |
  | `research/output-template.md` | `.agents/skills/cmd-j-paper-analysis/SKILL.md`, `.codex/prompts/j-paper-analysis.md` | nothing |

  The last two are a known gap, not a claim of coverage. They re-express their source rather than
  copying it verbatim, so the byte-comparison used for the first one does not transfer as-is.

## Reversal Conditions

A reference file genuinely needs a section in one tree that must not appear in the other, and
adapting the body under a shared heading is not enough to express the difference. A harness-specific
*topic* — not a harness-specific *wording* — is the signal. If that arrives, supersede this ADR
rather than adding an exception list to the test.

## Related Decisions
- [Thorough about quality dimensions, lazy about new scope](../workflow/code-ladder-vs-completeness.md)

## Amendment Log
| Date | Change | Reason | By |
|------|--------|--------|-----|
