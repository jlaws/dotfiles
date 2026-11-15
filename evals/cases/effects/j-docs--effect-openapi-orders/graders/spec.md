---
type: llm
focus: { source: file, path: openapi.yaml }
---

All claims are required.
PASS if openapi.yaml declares an OpenAPI 3.x document with paths for POST /v1/orders, GET /v1/orders, and GET /v1/orders/{id}.
PASS if POST /v1/orders takes an Idempotency-Key header, returns 201 with a Location header, and every error response refers to one shared error schema under components.
PASS if GET /v1/orders is paginated by cursor and returns next_cursor, and 404 is declared only where an id lookup can fail rather than on every operation.
FAIL if the examples use foo, bar, or placeholder values, or 409 is declared on a read-only operation.
