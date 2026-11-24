---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt, the history of main after the merge, contains both the commit "Bump" that was on main on origin and the commit "Rename README to rst" from the feature branch.
FAIL if the commit "Bump" is missing from log.txt.
