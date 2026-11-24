---
type: llm
focus: { source: file, path: draft/commands/j-standup.md }
---

All claims are required.
PASS if the frontmatter has name j-standup, a description, and an argument-hint for the optional author.
PASS if the body reads the author from $ARGUMENTS.
FAIL if the frontmatter sets a model key.
