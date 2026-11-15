---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if report.md is essentially one line saying there are no findings.
PASS if report.md lists no bullets of style nits, type-hint suggestions, or hypothetical overflow issues presented as findings.
FAIL if the report manufactures findings for add(), or runs to more than five lines.
