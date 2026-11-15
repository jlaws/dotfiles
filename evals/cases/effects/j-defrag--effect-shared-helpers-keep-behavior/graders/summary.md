---
type: llm
focus: { source: file, path: defrag-summary.md }
---

All claims are required.
PASS if defrag-summary.md reports the duplicated name normalization and the duplicated email validation under shop/ as fixed clusters.
PASS if defrag-summary.md lists the inconsistent invalid-email handling (customers return None, orders and invoices raise ValueError) as deferred or residual because converging it would change caller-visible behavior.
FAIL if defrag-summary.md claims that inconsistency was converged.
