---
type: llm
focus: { source: file, path: docs/cli.md }
---

All claims are required.
PASS if docs/cli.md says that without --output the tool writes a .jsonl file next to the input, named after the input file.
FAIL if docs/cli.md still says that without --output the tool prints to stdout.
