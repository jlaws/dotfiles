---
type: llm
focus: { source: file, path: doc-sync.md }
---

All claims are required.
PASS if doc-sync.md lists README.md as checked and states that --out was changed to --output.
PASS if every change it reports is consistent with the shipped behavior described in the request.
FAIL if it reports documenting any flag the request did not mention.
