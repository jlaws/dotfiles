---
type: llm
focus: { source: file, path: investigation.md }
---

All claims are required.
PASS if the root cause is that the README documents --out while tool.py defines --output, with file:line references for both.
PASS if the report classifies the fault as documentation (the README) rather than the tool's code, with a reason.
PASS if the report records the repro command and its observed error output from a run made before the fix, and a passing run after.
FAIL if the fix changes tool.py to accept --out without first stating that the documentation is what is wrong.
