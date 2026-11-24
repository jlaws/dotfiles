---
type: llm
focus: { source: file, path: merge-notes.md }
---

All claims are required.
PASS if the notes identify that, once merged, calc/stats.py breaks because it imports add, which main had renamed to plus (found by running the suite on the merged code), and say what was changed to fix it.
PASS if the notes report a passing test run on main after the merge and fix; a skipped weighted-total test is expected and does not count against it.
FAIL if the only test evidence the notes give is the feature branch's own run, made without main's rename in the tree.
