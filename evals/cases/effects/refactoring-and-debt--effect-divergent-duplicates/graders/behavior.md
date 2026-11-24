---
type: llm
focus: { source: file, path: calc/orders.py }
---

All claims are required.
PASS if validate_cart_line and validate_bulk_line both call one shared helper defined in calc/orders.py for the checks they have in common.
FAIL if the code shows that validate_cart_line(10) no longer returns None or validate_bulk_line(10) no longer returns "quantity must be at most 10", for example because both functions now apply the same upper-bound comparison. Unifying the lower bound as either qty < 1 or qty <= 0 is allowed: for whole numbers the two are the same, so it is not a behavior change.
