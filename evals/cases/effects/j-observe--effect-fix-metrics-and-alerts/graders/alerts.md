---
type: llm
focus: { source: file, path: obs/alerts.fixed.yml }
---

All claims are required.
PASS if the availability alerts compare error-budget burn rate against the 99.9% SLO using both a long and a short window combined with and, at a burn-rate threshold that is valid for that window pair.
PASS if every alert has a runbook link in its annotations.
PASS if every paging alert needs more than one evaluation to fire (for example a for: duration, or windows long enough that one bad scrape cannot trigger it).
FAIL if any rule still pages on a single 1-minute window.
