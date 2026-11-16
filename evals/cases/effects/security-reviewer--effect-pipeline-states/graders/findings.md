---
type: llm
focus: { source: file, path: security-report.md }
---

All claims are required.
PASS if security-report.md reports the string-built SQL in api/db.py find_orders as injection, and reports the shell=True command built from report_name in api/export.py archive as command injection, each with a file and line.
PASS if security-report.md reports the hardcoded PAYMENT_API_SECRET in api/config.py as exposed secret material.
PASS if security-report.md lists the parameterized find_order query and the list-argument ls call under rejected (or as not vulnerable) rather than as findings.
FAIL if security-report.md labels any lead confirmed, or presents severities as final rather than provisional.
