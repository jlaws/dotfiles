---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if README.md no longer mentions legacy-convert.sh or a Legacy wrapper section.
PASS if README.md still has its title, the Usage example with --input and --output, and the sentence that --input is required.
FAIL if the usage text or other unrelated lines were reworded.
