---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-docs write the API reference for our orders service. Save an OpenAPI 3 spec to ./openapi.yaml for POST /v1/orders (creates an order from a customer id and a list of items with sku and quantity; returns 201 with the order and a Location header) and GET /v1/orders/{id} (returns the order or 404). Requirements: the create endpoint is retryable safely through an Idempotency-Key header; document 400, 401, 404, 409, and 422 error responses only on the operations where they can occur, all using one shared error schema; list endpoints paginate with a cursor, so also add GET /v1/orders returning a page with a next_cursor; examples use realistic values, not foo or bar; and save a human-readable page for the create endpoint to ./create-order.md that shows a curl example first and both a success and an error response body.
