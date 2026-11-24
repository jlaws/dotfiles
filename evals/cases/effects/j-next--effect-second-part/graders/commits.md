---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt has exactly one commit whose message is 'Document the empty-list mean', and it sits above 'Return 0.0 for the mean of an empty list'.
PASS if 'Initial commit' and 'Return 0.0 for the mean of an empty list' each appear exactly once in log.txt.
FAIL if log.txt shows any new commit, other than 'Document the empty-list mean', whose message is not about a plan or progress-ledger update.
