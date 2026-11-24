---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md reports that median in calc/ops.py returns the upper middle element for an even-length list (median([1, 2, 3, 4]) gives 3, not 2.5), a function that exists only in the uncommitted change.
PASS if review.md reports that the uncommitted tests/test_median.py covers only an odd-length list (no even-length or empty-list case).
FAIL if review.md says mean([]) still raises ZeroDivisionError, which is true only of the committed code.
