---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if the report says the worktree was branched from origin/main, naming a commit.
PASS if the report explains the choice, such as that the local HEAD carries an unpushed commit that should not be in the new branch.
FAIL if the report says the branch started from the current HEAD.
