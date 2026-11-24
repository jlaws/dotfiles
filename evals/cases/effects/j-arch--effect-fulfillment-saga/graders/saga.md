---
type: llm
focus: { source: file, path: saga.md }
---

All claims are required.
PASS if saga.md coordinates the fulfillment flow by orchestration: one orchestrator service or workflow engine drives the five steps and tracks which have completed, rather than each service reacting to the previous service's event.
PASS if, when a step fails, saga.md undoes only the steps that completed, in reverse order of completion (for example, a failed courier booking voids the authorization and then releases the reservation).
FAIL if saga.md chooses choreography for the fulfillment flow, for example on the grounds that the platform already coordinates signup through Kafka events.
