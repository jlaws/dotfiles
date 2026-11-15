---
type: llm
focus: { source: file, path: proposed-doc-changes.diff }
---

All claims are required.
PASS if it is a unified diff against README.md that replaces --out with --output in the usage command and in the flag description.
FAIL if the diff rewrites unrelated lines, adds new sections, or restructures the README.
