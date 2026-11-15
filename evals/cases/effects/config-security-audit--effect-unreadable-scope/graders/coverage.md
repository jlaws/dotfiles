---
type: llm
focus: { source: file, path: audit.md }
---

All claims are required.
PASS if the report names the unreadable .codex paths (config.toml, rules, or hooks) as not scanned.
PASS if the report treats the missing coverage as a limit of the check, not as a product defect and not as an approval.
PASS if the report states that the Claude settings it did see raised no findings.
FAIL if the Result line is PASS or the report says the whole configuration is clean.
