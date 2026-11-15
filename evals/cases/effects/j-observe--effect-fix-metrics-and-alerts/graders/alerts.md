---
type: llm
focus: { source: file, path: obs/alerts.fixed.yml }
---

All claims are required.
PASS if the availability alerts use multi-window burn rate (a long window plus a short window combined with and) against the 99.9% budget, with a fast-burn rule near 14.4x on 1h and 5m and a slower rule that opens a ticket.
PASS if every alert has a runbook link in its annotations and the paging alerts set a for: duration of at least 2m.
PASS if the pod-restart alert is removed or demoted below critical, since it alerts on a cause rather than a symptom.
FAIL if any rule still pages on a single 1-minute window.
