---
type: llm
focus: { source: file, path: app/api/v1/endpoints/orders.py }
---

All claims are required.
PASS if the POST /orders handler in orders.py hands the order to a service function or class imported from another module, rather than creating the order itself.
PASS if orders.py makes no database calls: no select(), query(), execute(), add(), flush(), or commit() and no SQL text. Receiving a session through a dependency and passing it to the service is allowed.
FAIL if orders.py itself looks up whether the customer already has an unpaid order (a database query or a status comparison in the handler). Catching an error the service raises and turning it into the 409 response is allowed.
