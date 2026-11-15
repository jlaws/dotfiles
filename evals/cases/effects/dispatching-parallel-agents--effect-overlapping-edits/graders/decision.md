---
type: llm
focus: { source: file, path: decision.md }
---

All claims are required.
PASS if decision.md recognizes that both changes edit calc/ops.py and tests/test_ops.py, the same files.
PASS if the work was done by one agent, or by helpers run one after another or in isolated copies, and decision.md says so.
FAIL if decision.md describes two helpers editing calc/ops.py or tests/test_ops.py at the same time in one working tree.
