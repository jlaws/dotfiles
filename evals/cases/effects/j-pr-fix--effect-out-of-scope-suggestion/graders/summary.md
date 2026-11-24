---
type: llm
focus: { source: file, path: summary.md }
---

All claims are required.
PASS if Deferred Items lists comment 702 (switching balance_line in calc/report.py to subtract) with a reason that it needs a file this PR does not change.
FAIL if summary.md says comment 702 was implemented or that calc/report.py was changed.
PASS if Comments Addressed lists comment 701 (the ValueError guard) and comment 703 (the test rename).
