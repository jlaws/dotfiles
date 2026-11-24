---
type: llm
focus: { source: file, path: quality-review.md }
---

All claims are required.
PASS if it flags the retries parameter of fetch in app/http.py as an option that never varies, because every call site passes 3 or uses the default of 3, and proposes removing the parameter.
PASS if it flags BaseExporter in app/export.py as an abstraction with only one implementation (CsvExporter).
PASS if the timeout parameter of fetch, which takes 5, 30, and 10 at its call sites, is not reported as a parameter that never varies.
FAIL if the TODO(#412) in app/prices.py, which carries a ticket and a reason, is reported as a finding.
