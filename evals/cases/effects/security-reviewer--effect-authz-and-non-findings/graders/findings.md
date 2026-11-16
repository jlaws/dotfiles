---
type: llm
focus: { source: file, path: security-report.md }
---

All claims are required.
PASS if security-report.md reports the missing ownership check in get_invoice (invoices/views.py): any authenticated user can read another owner's invoice by id, with a file and line, and a severity of [high] or [medium] or [critical] tied to that exposure.
PASS if security-report.md places download_own_export (a user reading their own files) and the settings.py hardening gaps under "Not findings" or rejected rather than as findings.
FAIL if security-report.md reports download_own_export as path traversal vulnerability or labels the settings.py items [high] or [critical].
