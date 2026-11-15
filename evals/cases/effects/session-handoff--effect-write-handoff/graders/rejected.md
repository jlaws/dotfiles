---
type: llm
focus: { source: file, path: HANDOFF.md }
---

All claims are required.
PASS if HANDOFF.md records both rejected approaches (ValueError and None) with the reason each was dropped.
PASS if HANDOFF.md is under 100 lines.
FAIL if either rejected approach is missing its reason.
