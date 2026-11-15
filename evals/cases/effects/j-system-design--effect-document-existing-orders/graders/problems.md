---
type: llm
focus: { source: file, path: architecture.md }
---

All claims are required.
PASS if architecture.md identifies the circular dependency between src/orders.js and src/inventory.js.
PASS if architecture.md flags src/services.js as a module mixing unrelated responsibilities (payments, email, audit logging, formatting) and notes that placeOrder runs reserve, charge, and receipt with no failure handling or rollback.
PASS if architecture.md notes that audit logging writes the full order payload (card data included) to the console, as a risk.
FAIL if it proposes new features or components unrelated to fixing the found problems, or lacks a risks and open questions section.
