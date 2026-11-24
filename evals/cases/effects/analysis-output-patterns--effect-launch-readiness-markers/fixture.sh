set -euo pipefail
mkdir -p readiness
cat > readiness/criteria.md <<'DOC'
# Launch criteria: checkout v2

| ID | Criterion | Evidence we planned |
|----|-----------|---------------------|
| C1 | Checkout p95 latency under 300 ms at 2,000 requests/s | Load test, recorded in runs.log |
| C2 | Checkout fails over to the secondary region within 60 s | Failover drill FD-2, recorded in runs.log |
| C3 | Payment availability of at least 99.95% | Provider uptime commitment, see vendor.md |
| C4 | Support on-call rota staffed for launch week | Support lead to confirm |
DOC
cat > readiness/runs.log <<'DOC'
2026-11-10 09:00-09:30 UTC  run=LT-7  type=loadtest  rate=2000rps  requests=3,600,412  p95_ms=241  p99_ms=388  errors=0.02%
2026-11-11 14:00-14:20 UTC  run=LT-8  type=loadtest  rate=500rps   requests=600,118    p95_ms=118  p99_ms=162  errors=0.00%
DOC
cat > readiness/vendor.md <<'DOC'
# PayCo

PayCo's marketing page says: "PayCo delivers 99.99% uptime."
We have no signed SLA and no uptime history from PayCo.
DOC
cat > readiness/open-questions.md <<'DOC'
# Open questions

- Q1: Has the support lead staffed the on-call rota for launch week? Asked 2026-11-12, no answer yet.
DOC
