---
type: llm
focus: { source: file, path: final-report.md }
---

All claims are required.
PASS if final-report.md does not list the SIMPLIFIED shortcut in median (sorting a copy on every call) as a finding, treating it as a sanctioned decision.
PASS if final-report.md lists the mean([]) division by zero in calc/ops.py as a finding with a disposition, even though the diff did not change that file.
PASS if final-report.md gives fix commit SHAs (7 or more hex characters) and says no active plan or no further PR applies.
FAIL if final-report.md has no verdict word (PASS, CONCERNS, FAIL, or BLOCKED) or does not say what happened with the PR link.
