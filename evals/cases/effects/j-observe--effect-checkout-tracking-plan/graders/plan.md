---
type: llm
focus: { source: file, path: tracking-plan.md }
---

All claims are required.
PASS if tracking-plan.md states one naming convention (for example snake_case object_action) and every event and property follows it.
PASS if order placed and revenue are emitted server-side, with an explanation that client events can be dropped by ad blockers, while UI steps such as cart viewed may be client-side.
PASS if personal data is excluded through an enforcement mechanism such as a per-event property allowlist or middleware that strips unlisted fields, and no event property is an email, name, address, or IP.
FAIL if the plan lacks a review or schema process for preventing drift, or uses raw free-text or unbounded IDs as grouping properties.
