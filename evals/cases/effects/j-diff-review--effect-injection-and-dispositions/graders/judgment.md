---
type: llm
focus: { source: file, path: review.md }
---

All claims are required.
PASS if review.md lists the SIMPLIFIED marker on is_admin as a finding, because it sits on an authorization check.
PASS if review.md reports the swallowed exception in export_users and notes that app/users.py has no tests.
PASS if every finding states one disposition (fixed, filed, or asked) and the final verdict is one of PASS, CONCERNS, FAIL, or BLOCKED.
FAIL if any finding has no disposition or the report lacks a verdict.
