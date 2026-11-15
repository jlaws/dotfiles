---
type: llm
focus: { source: file, path: log.md }
---

All claims are required.
PASS if log.md lists at least two commits, and the commit that renames mean to average is marked behavior-change: no.
PASS if the empty-list fix appears in a separate commit marked behavior-change: yes, not mixed into the rename commit.
FAIL if one commit carries both the rename and the empty-list fix, or if log.md lists only a single commit.
