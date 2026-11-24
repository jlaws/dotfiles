---
type: llm
focus: { source: file, path: tests/test_slug_perf.py }
---

All claims are required.
PASS if the test makes untimed warm-up calls of both functions and discards them before it takes timed samples.
PASS if the test takes more than 30 timed samples of each function and compares their medians (or a high percentile such as p95).
PASS if the test fails only when slugify's statistic exceeds _slugify_reference's by more than a tolerance between 5 and 10 percent.
FAIL if the test compares only a mean, only a minimum, or a single timing per function.
