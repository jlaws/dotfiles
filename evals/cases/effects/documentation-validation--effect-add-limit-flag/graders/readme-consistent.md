---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if README.md documents the --limit option with a short description.
PASS if the README usage example uses the flag names that tool.py actually defines (--input and --output), correcting the stale --out if it was left.
FAIL if README.md still shows --out as a working flag.
