---
type: llm
focus: { source: file, path: prd.md }
---

All claims are required.
PASS if prd.md puts SSO and the analytics export in non-goals or a deferred list instead of in the proposed solution, noting that there is no evidence for them.
PASS if prd.md has success metrics with at least one leading and one lagging indicator, each with a baseline, a target, and a timeframe.
PASS if the acceptance criteria cover duplicate emails, an upload over 500 rows, and the flag-off case in Given/When/Then form.
PASS if the launch plan has stages (such as dogfood, a partial beta, then general availability) and rollback triggers with numeric thresholds (for example error rate above a multiple of baseline).
FAIL if prd.md builds SSO or the analytics export into the proposed solution.
