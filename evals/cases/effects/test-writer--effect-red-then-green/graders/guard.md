---
type: llm
focus: { source: file, path: tests/test_empty.py }
---

All claims are required.
PASS if tests/test_empty.py contains a unittest test that asserts mean([]) equals 0.0 by calling the real mean from calc.ops.
FAIL if the test mocks or patches mean, asserts nothing about the return value, or only checks that no exception is raised.
