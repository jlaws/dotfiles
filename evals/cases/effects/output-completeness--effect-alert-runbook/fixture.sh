set -euo pipefail
mkdir -p alerts
cat > alerts/alerts.yaml <<'EOF'
# Alerting rules for the orders service (namespace: orders, deployment: orders-api).
alerts:
  - name: HighErrorRate
    severity: page
    expr: 5xx responses above 2% of requests for 5m
    owner: orders-oncall
  - name: HighLatencyP95
    severity: page
    expr: p95 latency of /checkout above 800ms for 10m
    owner: orders-oncall
  - name: DiskUsageWarning
    severity: ticket
    expr: /var/lib/orders volume above 80% on any orders-db node
    owner: orders-oncall
  - name: DiskUsageCritical
    severity: page
    expr: /var/lib/orders volume above 95% on any orders-db node
    owner: orders-oncall, escalate to storage-team after 15m
  - name: MemoryPressure
    severity: ticket
    expr: orders-api container memory above 90% of its limit for 15m
    owner: orders-oncall
  - name: ReplicaLagHigh
    severity: page
    expr: orders-db replica more than 30s behind the primary for 5m
    owner: orders-oncall, escalate to dba-oncall after 10m
  - name: QueueBacklogWarning
    severity: ticket
    expr: fulfilment queue depth above 5000 messages for 10m
    owner: orders-oncall
  - name: QueueBacklogCritical
    severity: page
    expr: fulfilment queue depth above 50000 messages, or oldest message older than 30m
    owner: orders-oncall, escalate to fulfilment-team after 15m
  - name: CertExpiringSoon
    severity: ticket
    expr: TLS certificate for orders.example.invalid expires within 14 days
    owner: platform-team
    runbook: out of scope for this runbook; tracked in issue #88
EOF
