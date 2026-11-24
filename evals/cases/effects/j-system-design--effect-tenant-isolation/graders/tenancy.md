---
type: llm
focus: { source: file, path: architecture.md }
---

All claims are required.
PASS if the Free and Team tenants share a database, separated at the row level by tenant_id (optionally with row-level security or partitioning), rather than getting a schema or a database each.
PASS if each HIPAA Enterprise tenant gets dedicated infrastructure of its own: at least its own database and its own deployment or compute, not a schema or rows in the shared database.
PASS if the monthly finance report reads from a separate reporting path, with data copied or ETL'd from every tenant store into a warehouse or reporting database, not from queries that join across tenant databases or schemas.
FAIL if any HIPAA tenant's data stays in the shared database or in a schema inside it.
