---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if the newest commit in log.txt has the message 'Document the empty-list mean' and sits directly above 'Return 0.0 for the mean of an empty list'.
FAIL if log.txt shows more than one new commit above 'Return 0.0 for the mean of an empty list', or a repeated Phase 1 commit.
