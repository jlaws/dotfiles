---
type: llm
focus: { source: file, path: fix-plan.md }
---

All claims are required.
PASS if comment 201 is labelled blocking and ordered before the suggestion 202.
PASS if comment 203 is in the UNCLEAR list with an explanation of what is unclear.
FAIL if 203 is given a concrete code fix in the plan.
