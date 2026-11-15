---
type: llm
focus: { source: file, path: audit.md }
---

All claims are required.
PASS if audit.md states a non-zero (failing) exit status for the check.
PASS if audit.md lists .claude/skills/broken-skill/SKILL.md as a failure because its frontmatter has no description.
PASS if audit.md separates warnings from failures and includes the missing argument-hint warning for .claude/commands/j-demo.md.
FAIL if audit.md reports the tree as clean or gives no exit status.
