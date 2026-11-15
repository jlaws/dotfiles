---
type: llm
focus: { source: file, path: tests/test_convert.py }
---

All claims are required.
PASS if the mock of RateClient is constrained to the real class interface (spec, spec_set, or autospec), so a call to a method the class does not have would fail.
PASS if the mocked fetch_rate returns a realistic numeric rate (a float such as 0.92), not a placeholder string or an unconfigured Mock.
PASS if the tests assert on the value convert returns, including a rounding case and the ValueError case.
FAIL if a test passes a bare Mock() or MagicMock() with no spec as the client.
