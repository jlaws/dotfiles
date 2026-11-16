---
type: llm
focus: { source: file, path: design.md }
---

All claims are required.
PASS if design.md interprets what undo means for the existing calc package (add and mean are stateless functions) and picks one interpretation with a reason.
PASS if every open question in design.md is answered with a decision and reason, and no unresolved open-questions section remains.
PASS if the numbered assumptions are falsifiable claims about usage or the code, not vague wishes.
FAIL if design.md lists fewer than two concrete out-of-scope items or proposes edits in this session to files other than design.md.
