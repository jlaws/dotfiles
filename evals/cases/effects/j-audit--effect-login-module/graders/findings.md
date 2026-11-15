---
type: llm
focus: { source: file, path: audit.md }
---

All claims are required.
PASS if audit.md reports the SQL injection in login (string-formatted username in the query).
PASS if audit.md reports the shell command injection in export_report and the weak MD5 password hashing.
PASS if audit.md does not list get_profile as a vulnerability (its query is parameterized).
FAIL if audit.md lacks a threat-model section before the findings, or any finding is missing a STRIDE category or a severity.
