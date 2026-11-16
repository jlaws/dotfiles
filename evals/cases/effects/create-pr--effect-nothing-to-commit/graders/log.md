---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt lists exactly two commits: "Add double" and "Initial commit", and no others.
FAIL if log.txt shows an extra commit (for example an empty or placeholder commit).
