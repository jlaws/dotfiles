---
type: llm
focus: { source: file, path: resolution.md }
---

All claims are required.
PASS if resolution.md names the commit "Tidy ops helpers (#20)" as the one that introduced the regression.
PASS if it gives evidence tied to that commit: bisect output, or mean([]) returning 0.0 at that commit's parent and raising ZeroDivisionError at that commit.
PASS if it gives steps a reviewer can run to verify the fix and a rollback plan.
FAIL if it names "Format ops.py (#23)" as the commit that introduced the regression; that commit only reformatted the line.
