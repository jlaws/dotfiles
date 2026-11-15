---
type: llm
focus: { source: file, path: audit.md }
---

All claims are required.
PASS if the report flags that settings.local.json is not gitignored.
PASS if the final Result line is `Result: FAIL`.
FAIL if the final Result line is PASS or CONCERNS.
