---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt lists exactly two commits: the initial commit and one whose message is 'Return 0.0 for the mean of an empty list'.
FAIL if log.txt contains a commit about documenting the empty-list mean, or more than two commits.
