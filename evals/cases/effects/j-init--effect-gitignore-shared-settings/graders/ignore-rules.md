---
type: llm
focus: { source: file, path: .gitignore }
---

All claims are required.
PASS if .gitignore ignores .claude/settings.local.json, by name or through a pattern such as *.local.json.
PASS if .gitignore ignores the .env file.
FAIL if .gitignore would keep .claude/settings.json out of git: a .claude/ directory rule always does, and a .claude/* rule does unless a later !.claude/settings.json line re-includes the file.
