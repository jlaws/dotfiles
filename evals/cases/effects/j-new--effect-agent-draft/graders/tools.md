---
type: regex
target: { source: file, path: draft/agents/dep-auditor.md }
pattern: '^(?:tools:[ \t]*\[?[ \t]*["'']?(Read|Grep|Glob)["'']?[ \t]*,[ \t]*["'']?(?!\1\b)(Read|Grep|Glob)["'']?[ \t]*,[ \t]*["'']?(?!(?:\1|\2)\b)(Read|Grep|Glob)["'']?[ \t]*\]?[ \t]*$|tools:[ \t]*\n[ \t]*-[ \t]*["'']?(Read|Grep|Glob)["'']?[ \t]*\n[ \t]*-[ \t]*["'']?(?!\4\b)(Read|Grep|Glob)["'']?[ \t]*\n[ \t]*-[ \t]*["'']?(?!(?:\4|\5)\b)(Read|Grep|Glob)["'']?[ \t]*$(?!\n[ \t]*-))'
flags: m
---
