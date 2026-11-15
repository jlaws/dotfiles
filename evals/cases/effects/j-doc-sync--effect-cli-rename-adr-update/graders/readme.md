---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if the README usage example and flag description use --output and no longer use --out as the flag.
PASS if the README does not mention a --format flag and still keeps its original title and its statement that --input is required.
FAIL if the README was rewritten wholesale rather than edited in the stale lines.
