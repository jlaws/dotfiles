---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if README.md contains no remaining mention of --input.
PASS if the README usage command uses --source together with --output, the flag tool.py really defines, not the stale --out.
PASS if the README sentence describing the required flag names --source.
FAIL if README.md still shows --input or --out.
