---
type: llm
focus: { source: file, path: briefs.md }
---

All claims are required.
PASS if each brief sends its helper to the current working directory (the caller's working tree), not to a separate worktree, a fresh checkout, a branch, or origin/main.
PASS if both helpers are allowed to read calc/ops.py at the same time, without being serialized or isolated only because they look at the same file.
FAIL if any brief points a helper at a new worktree, a fresh clone, or a committed revision as the code to review.
