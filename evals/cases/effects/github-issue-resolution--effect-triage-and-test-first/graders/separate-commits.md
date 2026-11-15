---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt lists at least two commits after the initial commit, one for the test and one for the fix.
PASS if the commits after the initial one reference #57.
FAIL if the test and the fix share a single commit.
