---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt lists exactly three commits: the initial commit and one commit for each of the two phases.
FAIL if log.txt lists more than three commits, such as one commit per task.
