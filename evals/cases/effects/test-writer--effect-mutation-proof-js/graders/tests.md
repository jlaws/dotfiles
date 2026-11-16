---
type: llm
focus: { source: file, path: test/complete.test.js }
---

All claims are required.
PASS if test/complete.test.js imports complete from ../src/todo.js and has separate tests for marking the matching item done, leaving others untouched, not mutating the input, and an unknown id returning an equal list.
PASS if the no-mutation test would fail against an in-place implementation: it freezes the input list and items (Object.freeze) or compares the original objects after the call.
FAIL if any test only asserts that complete returns something, or the tests assert nothing about the returned items' done values.
