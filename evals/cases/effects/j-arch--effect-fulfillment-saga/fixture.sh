set -euo pipefail
mkdir -p docs
cat > docs/architecture.md <<'MD'
# Platform architecture

Services: storefront-api, inventory, payments, shipping, loyalty, and mailer. Each service owns its
database; there is no shared database. Services talk through Kafka topics.

## Existing flow: signup

1. storefront-api writes the user and publishes `UserSignedUp`.
2. mailer consumes `UserSignedUp` and sends the welcome email.

If the email fails, mailer retries it; the user record stays. This flow has run without incident
for two years.
MD
cat > docs/fulfillment-flow.md <<'MD'
# Order fulfillment (new)

When a customer places an order, every step below must succeed, or the order is cancelled.

| Step | Service | Action | Undo |
|---|---|---|---|
| 1 | inventory | reserve the items | release the reservation |
| 2 | payments | authorize the card | void the authorization |
| 3 | shipping | book a courier pickup | cancel the pickup |
| 4 | loyalty | award points for the order | revoke the points |
| 5 | payments | capture the authorized amount | refund the capture |

Any step can fail.
MD
