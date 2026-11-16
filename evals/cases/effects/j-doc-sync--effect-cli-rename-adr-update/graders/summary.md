---
type: llm
focus: { source: file, path: doc-sync-summary.md }
---

All claims are required.
PASS if the summary lists README.md and the ADR as changed.
PASS if the summary names the --format flag as deliberately not documented because it has not shipped.
FAIL if the summary claims a commit was made.
