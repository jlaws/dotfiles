---
type: llm
focus: { source: file, path: retro.md }
---

All claims are required.
PASS if retro.md reports 9 commits for 2026-11-01 to 2026-11-14 and 4 commits for the previous period, with an upward trend.
PASS if retro.md reports 23 lines added and 5 lines removed for the current period, and 11 added and 0 removed for the previous period.
PASS if retro.md identifies the revert of the power feature as work that did not stick and lists calc/ops.py as the high-churn file, worded as possible instability rather than a confirmed problem.
FAIL if retro.md contains a per-author breakdown or uses numbers that differ from these.
