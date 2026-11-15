---
type: llm
focus: { source: file, path: notes.md }
---

All claims are required.
PASS if notes.md states that complete() had no test coverage before the refactor.
PASS if notes.md says tests pinning the existing behavior of complete() were written before the structural change.
FAIL if notes.md claims behavior was preserved without naming a test or run that exercised complete().
