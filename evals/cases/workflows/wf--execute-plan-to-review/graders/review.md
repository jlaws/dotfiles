---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md gives one closing verdict for the review: PASS, CONCERNS, FAIL, or BLOCKED.
PASS if review.md says that no part of the plan is left for a later pull request (the plan is complete).
FAIL if review.md lists a finding without saying how it was handled (fixed, filed, or asked).
