---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if the usage example and the flag description use --output, with no remaining --out.
PASS if the README does not mention --delimiter or TSV input.
PASS if the title, the one-line description 'Convert a CSV file to JSON lines.', and the section headings are unchanged and no new sections were added.
FAIL if the README gained any text about unshipped behavior.
