---
type: llm
focus: { source: file, path: tests/test_median.py }
---

All claims are required.
PASS if the tests cover an odd-length list, an even-length list whose result is the average of the two middle values (a literal such as 2.5), and an empty list raising ValueError.
PASS if the tests check that the input list is left unchanged after the call, using an unsorted input.
FAIL if an expected value is computed by calling median itself or by reimplementing it in the test.
