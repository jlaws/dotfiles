---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if report.md says there was nothing to commit and says what was checked (the working tree status was clean on the branch).
FAIL if report.md claims to have committed, pushed new work, or opened a pull request.
