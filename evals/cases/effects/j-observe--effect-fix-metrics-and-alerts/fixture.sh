set -euo pipefail
mkdir -p obs
cat > obs/metrics.js <<'JS'
import client from "prom-client";

export const latency = new client.Summary({
  name: "http_request_duration_seconds",
  help: "Request latency",
  labelNames: ["route", "user_id"],
  percentiles: [0.5, 0.9, 0.99],
});

export const requests = new client.Counter({
  name: "http_requests_total",
  help: "Requests",
  labelNames: ["route", "status", "request_id"],
});

export function observe(req, res, seconds) {
  latency.labels(req.route, req.user.id).observe(seconds);
  requests.labels(req.route, String(res.statusCode), req.id).inc();
}
JS
cat > obs/alerts.yml <<'YML'
groups:
  - name: orders
    rules:
      - alert: OrdersHighErrorRate
        expr: sum(rate(http_requests_total{status="500"}[1m])) / sum(rate(http_requests_total[1m])) > 0.001
        labels:
          severity: critical
        annotations:
          summary: Orders error rate above 0.1%
      - alert: OrdersPodRestarted
        expr: increase(kube_pod_container_status_restarts_total{pod="orders-0"}[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: An orders pod restarted
YML
