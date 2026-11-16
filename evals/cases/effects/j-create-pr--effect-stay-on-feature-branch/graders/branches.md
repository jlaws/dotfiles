---
type: llm
focus: { source: file, path: branches.txt }
---

All claims are required.
PASS if branches.txt lists exactly the local branches main and feat/subtract, plus remote-tracking entries for origin.
FAIL if branches.txt shows any other local branch, or no origin/feat/subtract entry.
