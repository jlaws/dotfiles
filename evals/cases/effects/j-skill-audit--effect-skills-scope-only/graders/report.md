---
type: llm
focus: { source: file, path: audit-report.md }
---

All claims are required.
PASS if the report opens with failure and warning counts from the mechanical check and reports the broken-skill SKILL.md as missing its description, with a fix.
PASS if the report does not list any finding about the j-demo command, .claude/CLAUDE.md, or .claude/settings.json.
PASS if the report ends with exactly one named highest-value next change.
FAIL if the report lists findings about the j-demo command, .claude/CLAUDE.md, or .claude/settings.json.
