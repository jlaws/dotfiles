---
type: llm
focus: { source: file, path: security-report.md }
---

All claims are required.
PASS if security-report.md reports the missing ownership check in get_invoice (invoices/views.py): any authenticated user can read another owner's invoice by id, with a file and line, and a severity of [high] or [medium] or [critical] tied to that exposure.
PASS if security-report.md places download_own_export in invoices/cli.py (a local tool reading the caller's own files) and both settings.py values under "Not findings" rather than as findings.
FAIL if security-report.md reports download_own_export as a path traversal finding, or reports either settings.py value (ALLOWED_HOSTS or SESSION_COOKIE_SAMESITE) as a finding with a severity label.
