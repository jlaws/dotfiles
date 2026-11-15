---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if the report has a raw terminal output code block for each of Tasks 1, 2 and 3, labelled by task.
PASS if the Task 1 output shows the new test failing before the fix, and the Task 2 output shows no failures.
PASS if the report ends with the output of a final full-suite run after the last task.
FAIL if results are described in prose without the pasted output.
