---
type: llm
focus: { source: file, path: calc/ops.py }
---

All claims are required.
PASS if mean returns 0.0 for an empty list, by any correct construction (an early return, a conditional expression, a max(len(xs), 1) divisor, or similar), and still returns the arithmetic mean for a non-empty list.
FAIL if mean([]) raises (ZeroDivisionError, ValueError, or any other exception) or returns anything other than 0.0, such as None or 0.
