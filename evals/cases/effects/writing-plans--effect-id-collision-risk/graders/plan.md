---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the plan has a Risks table in which every row names the file and line, test, or measurement that closes the risk.
PASS if the plan identifies that addTodo derives a new id from list.length + 1, so ids can collide after a removal, and decides the fix (for example max id + 1) with a test that adds an item after removing one.
PASS if each test is named with the assertion it makes, and the verification commands (such as `node --test`) come with expected output.
FAIL if the plan contains TBD, 'choose an appropriate', 'handle edge cases', or a follow-ups or known-limitations section.
