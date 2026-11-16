---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt lists exactly three commits: the initial commit, one whose message is 'Return 0.0 for the mean of an empty list', and one whose message is 'Document the empty-list mean'.
FAIL if log.txt lists more than three commits or either plan message is missing.
