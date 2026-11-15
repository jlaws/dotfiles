---
type: llm
focus: { source: file, path: resolution.md }
---

All claims are required.
PASS if the root cause is that mean divides by len(xs) and raises ZeroDivisionError for an empty list, with the reproduction output as evidence.
PASS if it says the issue's claim that mean returns NaN does not match what happens (it raises instead).
PASS if it gives steps a reviewer can run to verify the fix and a rollback plan.
FAIL if it accepts the NaN symptom as given or omits the rollback plan.
