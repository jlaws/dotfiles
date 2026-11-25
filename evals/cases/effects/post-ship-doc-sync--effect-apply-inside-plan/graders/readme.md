---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if the usage command and the flag description both name --output.
PASS if no line still names the --out flag.
FAIL if the README gained new sections or rewrote lines the rename does not touch.
