---
type: llm
focus: { source: file, path: obs/metrics.fixed.js }
---

All claims are required.
PASS if latency is a Histogram (not a Summary), so percentiles aggregate across instances.
PASS if neither user_id nor request_id remains as a label on any metric.
PASS if the instrumentation skips health-check or metrics routes (for example a guard on /health and /metrics) before recording.
FAIL if any unbounded value (user id, request id) is still used as a label.
