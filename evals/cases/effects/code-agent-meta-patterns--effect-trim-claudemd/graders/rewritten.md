---
type: llm
focus: { source: file, path: CLAUDE.md }
---

All claims are required.
PASS if CLAUDE.md contains no timestamp, "last updated" line, test count, coverage percentage, or open-issue count.
PASS if CLAUDE.md keeps the non-obvious constraints (immutable invoices with credit-note corrections, only ledger/ writes the entries table, BILLING_TEST_DB footgun) and drops the generic advice (clean code, write tests, Python history).
PASS if CLAUDE.md links to docs/openapi.yaml instead of restating the endpoint list.
PASS if CLAUDE.md has a section named Compact instructions (or equivalent heading containing "Compact") that says what must survive compaction.
PASS if each command is listed with a quiet flag or quiet form and the line a healthy run prints.
FAIL if the file keeps volatile counts or the generic advice.
