---
type: llm
focus: { source: file, path: obs/metrics.fixed.js }
---

All claims are required.
PASS if latency is a Histogram (not a Summary), so percentiles aggregate across instances.
PASS if neither user_id nor request_id remains as a label on any metric.
PASS if health-check and metrics routes are kept out of the SLI either in the instrumentation (for example a guard on /health and /metrics before recording) or in the SLI and alert queries, which this file allows by recording the route as a bounded label those queries can filter on.
FAIL if any unbounded value (user id, request id) is still used as a label.
