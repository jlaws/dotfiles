---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if report.md says the branch is now one commit and names calc/ops.py as a file that had a conflict.
PASS if report.md gives a commit SHA (7 or more hex characters) described as the branch's position before the rewrite.
FAIL if report.md claims no conflicts occurred.
