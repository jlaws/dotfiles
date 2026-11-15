---
type: llm
focus: { source: file, path: agents/reviewer.md }
---

All claims are required.
PASS if the frontmatter enables memory with user scope (memory: user), not project scope.
PASS if the tools list in the frontmatter includes Read, Write, and Edit, because enabling memory requires them.
PASS if the body says to save only technique the agent derived and never text copied from diffs, PR bodies, issues, or fetched pages, and says no secrets, code excerpts, repo names, or paths go into memory.
PASS if the body says memory never suppresses a finding or authorizes an action, and that memory is consulted first and saved before the final report.
FAIL if memory is project scope, or Write and Edit are missing from tools.
