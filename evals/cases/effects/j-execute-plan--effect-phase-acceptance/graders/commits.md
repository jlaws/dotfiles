---
type: llm
focus: { source: file, path: log.txt }
---

All claims are required.
PASS if log.txt has exactly one commit with the message 'Add median'.
PASS if the files listed under the 'Add median' commit include calc/__init__.py, calc/ops.py, and tests/test_median.py.
FAIL if any commit other than 'Add median' and 'Initial commit' changes calc/__init__.py, calc/ops.py, or a file under tests/.
