---
type: llm
focus: { source: file, path: cancel-spec.md }
---

All claims are required.
PASS if every resource row in the cleanup inventory (the payment hold, reserved inventory, and shipping label at minimum) has a non-blank Destroyed By and a timeout or auto-release.
PASS if the partial failure branch compensates completed steps in reverse order of creation (for example void the label, release inventory, release the hold) and reports the partial state.
PASS if the state table has an outbound transition for every non-terminal state, and the terminal states are named.
PASS if the handoff contract names an idempotency key, a maximum retry count with backoff, a timeout budget, and a dead letter destination.
FAIL if any resource in the cleanup inventory has a blank Destroyed By.
