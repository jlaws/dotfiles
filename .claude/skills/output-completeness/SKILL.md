---
name: output-completeness
description: "Use when output risks truncation, stubs, or omitted sections."
---

# Output Completeness

**A partial output is a broken output.** If you started it, finish it. When generating something
large, completeness beats brevity; a half-written module or a report with a stubbed section costs the
reader more than a longer answer would have.

## What incompleteness looks like

The failure is rarely a decision to stop; it is drifting into a placeholder because the remaining work
felt implied. In generated code that shows up as `// ...`, `/* rest of implementation */`, a function
body left as `pass`, or a class whose later methods were never written. In prose it shows up as "as
mentioned above", "similar to the above", a heading followed by "(Content TBD)", or a response that
simply ends mid-list.

None of these are wrong because the characters are forbidden. A `// TODO(#123)` marking work a plan
explicitly put out of scope is correct and useful. The test is whether the reader can use what you
produced, or has to come back for the rest.

Before responding to a large request, re-read it and confirm every deliverable you named is actually
present.

## When output genuinely will not fit

In a session that can write files and owns the artifact, write it to a file, in as many edits as
it takes, and report the path; a file has no response limit. A dispatched agent returns what is complete and names each missing section and why, because
nobody can reply "continue" to it. Only in a chat reply with nowhere to write, stop at a clean
breakpoint (the end of a function or a section, never mid-sentence) and say what is left. Silent
truncation and placeholder characters are never a handoff, because the reader cannot tell what is
missing.

## Load this when

The task produces something substantial and "partially done" would block the reader: a feature
spanning multiple files, full documentation sections, a multi-part refactor, a research analysis, or a
test suite where skeleton tests would give false confidence.

## Related

- `verification-before-completion`: how to weigh and report evidence. This skill governs whether the
  output is whole; that one governs how you characterize it.
- `${CLAUDE_SKILL_DIR}/../../references/workflow/llm-output-completeness.md`: root-cause research and parameter tuning for truncation
- `${CLAUDE_SKILL_DIR}/../../references/workflow/completeness-principle.md`: project-level completeness standards
