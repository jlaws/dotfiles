---
type: llm
focus: { source: file, path: plans/plan.md }
---

All claims are required.
PASS if the first phase creates a branch off origin/main and writes README.md documentation for percentile before any implementation phase.
PASS if at least one implementation phase writes a failing test before the code and ends in a single commit.
PASS if the plan states its pull request boundaries (one PR or a phase-to-PR mapping).
PASS if the final phase is a validation gate naming python3 -m unittest as the gate command or stating explicitly that no benchmark applies with a reason.
FAIL if the plan has an open-questions section, or a phase adds capabilities the spec did not ask for (for example interpolation modes or other statistics).
