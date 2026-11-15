---
type: llm
focus: { source: file, path: calc-explainer.html }
---

All claims are required.
PASS if the page states that mean's docstring promises 0.0 for an empty list.
PASS if the page states that the code instead divides by len(xs) and so fails (ZeroDivisionError) on an empty list.
FAIL if the page presents the empty-list mean as already returning 0.0.
PASS if the SVG diagram labels the actual functions add and mean from calc/ops.py.
