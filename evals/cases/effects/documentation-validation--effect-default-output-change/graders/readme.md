---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if README.md says that without --output the tool writes a .jsonl file next to the input, named after the input file.
FAIL if README.md still says that without an output flag the tool prints to stdout.
