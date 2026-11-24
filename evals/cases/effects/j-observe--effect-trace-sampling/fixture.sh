set -euo pipefail
mkdir -p docs
cat > docs/services.md <<'MD'
# Checkout tracing

- Six services take part in a checkout: edge-api, cart, pricing, inventory, payments, and orders.
  All use the OpenTelemetry SDK and export OTLP to one OpenTelemetry Collector.
- Peak production traffic: edge-api takes 450 requests per second, and no service exceeds 600
  requests per second.
- Every environment (dev, staging, production) samples 100% of traces today.
- The tracing backend bill tripled last quarter, and finance wants it cut.
- About 0.4% of checkout requests fail, and the slowest 1% take over 2 seconds. Those are the
  traces the on-call engineers open.
MD
