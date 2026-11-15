---
type: llm
focus: { source: file, path: investigation.md }
---

All claims are required.
PASS if the Root Cause section places the fault in mean() in calc/ops.py, dividing by len(xs) when xs is empty, and says the test expectation (0.0 for an empty list) is correct and matches the docstring.
PASS if Hypotheses Tested lists at least three hypotheses, each explicitly marked confirmed or refuted with evidence, and at most one is confirmed.
PASS if Verification names the command run and shows the failing test passing, with the rest of the suite still passing.
FAIL if the report proposes changing the test, skipping it, or catching the error in the test instead of fixing mean().
