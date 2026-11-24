---
type: llm
focus: { source: file, path: pr-body.md }
---

All claims are required.
PASS if the bullet points of what changed in pr-body.md include the median function added to calc/ops.py by the branch's earlier "Add median" commit.
PASS if the bullet points of what changed in pr-body.md include the median tests and the README note committed in this run.
FAIL if pr-body.md has no Test Plan section or no Merge Danger section.
