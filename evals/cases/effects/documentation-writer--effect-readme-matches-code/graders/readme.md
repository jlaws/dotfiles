---
type: llm
focus: { source: file, path: README.md }
---

All claims are required.
PASS if README.md no longer documents a --out flag; every flag shown is --input or --output.
PASS if README.md has a Behavior paragraph in prose stating that each CSV row becomes one JSON object per line keyed by the header, that all values are strings (no type conversion), and that an existing output file is overwritten.
FAIL if README.md uses stock phrases such as seamless, robust, leverage, delve, or "it is important to note", or the Behavior section is a bullet list.
