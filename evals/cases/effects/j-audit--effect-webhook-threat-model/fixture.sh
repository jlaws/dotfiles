set -euo pipefail
mkdir -p docs
cat > docs/webhook-design.md <<'MD'
# Payment webhook receiver

The payment provider POSTs JSON events to `/hooks/payments` on our API.

- The receiver verifies an HMAC-SHA256 signature header computed over the raw body with a shared secret.
- The event carries `id`, `type`, `amount`, `customer_email`, and `card_last4`.
- There is no timestamp in the signed payload, and no record of event ids already processed.
- Every request body is written to the application log at INFO level for debugging.
- On a `payment.succeeded` event the handler marks the order paid and enqueues fulfilment.
- The endpoint is public and is not rate limited. Failed handling returns 500 and the provider retries for 3 days.
- The shared secret lives in an environment variable and has never been rotated.
MD
