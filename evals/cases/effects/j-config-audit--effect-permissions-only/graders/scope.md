---
type: llm
focus: { source: file, path: permissions-audit.md }
---

All claims are required.
PASS if permissions-audit.md flags the blanket Bash(*) allow in .claude/settings.json and the reviewer agent's Write and Edit tools.
PASS if permissions-audit.md flags the hook running a script that pipes remote content to a shell.
FAIL if permissions-audit.md reports the AWS access key or any other secret as a finding, or contains an injection-vector finding.
