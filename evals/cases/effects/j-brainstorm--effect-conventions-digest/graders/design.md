---
type: llm
focus: { source: file, path: design.md }
---

All claims are required.
PASS if design.md cites CV-1, CV-2 and CV-3 by ID, each with a statement of how the design conforms.
PASS if design.md specifies that old equal to 0 raises calc.errors.CalcError (per CV-2), not ZeroDivisionError or a bare ValueError, and does no rounding inside the function (per CV-3).
PASS if design.md has a numbered list of falsifiable assumptions and compares at least two approaches with one recommended.
FAIL if design.md invents conventions that are not in docs/conventions.md or contradicts one of them.
