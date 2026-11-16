---
type: llm
focus: { source: file, path: doc-sync-summary.md }
---

All claims are required.
PASS if the summary says docs/adr/cli/wrapper-entrypoint.md was deleted, with the reason that the decision it records (shipping legacy-convert.sh) no longer exists.
PASS if the summary says the Legacy wrapper section was removed from README.md.
FAIL if the summary says the ADR was kept or rewritten, or claims a commit was made.
