---
type: llm
focus: { source: file, path: status.txt }
---

All claims are required.
PASS if status.txt shows the staged rename of README.md to NOTES.md still staged (a line starting with R).
FAIL if status.txt shows an untracked worktree directory such as '?? .worktrees/'.
