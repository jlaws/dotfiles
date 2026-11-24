---
type: llm
focus: { source: file, path: sync-summary.md }
---

All claims are required.
PASS if the summary lists README.md and docs/adr/cli/output-flag-name.md as changed, each with a reason tied to the --out to --output rename.
FAIL if the summary reports changing docs/adr/io/csv-dialect.md, which the rename does not affect.
