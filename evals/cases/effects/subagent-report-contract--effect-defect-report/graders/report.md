---
type: llm
focus: { source: file, path: report.md }
---

All claims are required.
PASS if the first line is a one-line verdict containing finding totals or a severity count.
PASS if the report includes the exact text ZeroDivisionError: division by zero from reproducing mean([]), unparaphrased.
PASS if the report cites the mean() division in calc/ops.py (the mean function body), and also lists the missing empty-list test coverage as a separate finding.
PASS if the report contains no narration of the search process (no sentences like "I looked at" or "First I read").
FAIL if the exact error text is paraphrased or a finding has no path:line.
