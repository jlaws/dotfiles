---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md opens with mechanical counts that include 2 failures (the missing description in broken-skill and the missing .claude/CLAUDE.md).
PASS if review.md rates all five axes (groundedness, coverage, freshness, structure, evidence) each with a reason.
PASS if review.md names exactly one weakest axis and exactly one next change.
PASS if each judgment finding cites a file and quotes text from it.
FAIL if the opening counts are invented or do not come from running the check.
