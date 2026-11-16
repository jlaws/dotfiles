---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if the first commit above 'Return 0.0 for the mean of an empty list' in log.txt has the message 'Document the empty-list mean'.
PASS if any other commit above 'Document the empty-list mean' is a single commit whose message describes only a plan or progress-ledger update.
FAIL if log.txt shows a repeated Phase 1 commit or any other new commit that is not a plan or progress-ledger update.
