---
type: llm
focus: { source: file, path: calc/orders.py }
---

All claims are required.
PASS if validate_cart_line and validate_bulk_line both call one shared helper defined in calc/orders.py for the checks they have in common.
PASS if, reading the code, validate_cart_line(10) still returns None and validate_bulk_line(10) still returns "quantity must be at most 10" (for example the helper takes the upper limit or its inclusiveness as a parameter, or one function keeps its own upper-bound check).
FAIL if both functions now apply the same upper-bound comparison, so one of them gives a different answer for a quantity of exactly 10 than the original code did.
PASS if the lower-bound check is unified as either qty < 1 or qty <= 0; for whole numbers these are the same, so unifying them is not a behavior change.
