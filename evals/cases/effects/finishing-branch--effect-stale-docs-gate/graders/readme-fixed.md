---
type: llm
focus: { source: file, path: pushed-readme.md }
---

All claims are required.
PASS if pushed-readme.md, the README as pushed to the remote branch, documents --output as the flag that names the output file.
FAIL if pushed-readme.md still shows --out as a working flag in its usage example or flag description.
