---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt shows exactly three commits: "Initial commit", "Add median", and one new commit made in this run.
FAIL if the "Add median" commit is missing, reworded, or folded into another commit (amended or squashed).
