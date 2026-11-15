---
type: llm
focus: { source: file, path: proof.md }
---

All claims are required.
PASS if the log has a separate entry for each of the three behaviors (marks done, does not mutate the input, ignores an unknown id) naming a specific mutation of complete() for each.
PASS if each entry records an observed failure from a real run (a failing test name or assertion message), not a prediction of what would fail.
PASS if the log shows a final run where the full suite passes after the restore.
FAIL if the same single mutation is reused for every test and the log does not show the individual test turning red, or if any entry only asserts the test would fail without showing it.
