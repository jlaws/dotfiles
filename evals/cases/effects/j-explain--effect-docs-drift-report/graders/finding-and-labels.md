---
type: llm
focus: { source: file, path: drift-report.html }
---

All claims are required.
PASS if the report finds that the README documents --out but tool.py defines --output.
PASS if the report says --out is rejected (unrecognized arguments) because allow_abbrev=False, citing tool.py as evidence.
PASS if statements are explicitly labeled evidence or inference, with the rejection of --out as evidence and the effect on users following the README as inference.
FAIL if background about what the tool does appears before the headline finding.
