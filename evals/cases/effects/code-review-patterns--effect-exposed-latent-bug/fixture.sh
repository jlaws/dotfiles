set -euo pipefail
mkdir -p billing
cat > billing/__init__.py <<'PY'
PY
cat > billing/money.py <<'PY'
"""Money helpers."""


def to_cents(amount):
    """Convert a dollar amount to integer cents."""
    return int(amount * 100)
PY
cat > billing/plans.py <<'PY'
"""Plan prices are whole dollars."""

PLANS = {"basic": 10, "team": 49, "business": 199}
PY
cat > billing/subscribe.py <<'PY'
from billing.money import to_cents
from billing.plans import PLANS


def charge_for(plan_name):
    return to_cents(PLANS[plan_name])
PY
cat > billing/refund.py <<'PY'
import logging

from billing.money import to_cents

log = logging.getLogger(__name__)


def refund_full(order):
    log.info("refund for %s", order.customer_email)
    return order.paid_cents


def refund_partial(order, amount_text):
    """Refund part of an order; amount_text is what the agent typed, such as "19.99"."""
    cents = to_cents(float(amount_text))
    if cents > order.paid_cents:
        raise ValueError("refund exceeds payment")
    return cents
PY
cat > change.diff <<'DIFF'
--- a/billing/refund.py
+++ b/billing/refund.py
@@ -1,8 +1,18 @@
 import logging

+from billing.money import to_cents
+
 log = logging.getLogger(__name__)


 def refund_full(order):
     log.info("refund for %s", order.customer_email)
     return order.paid_cents
+
+
+def refund_partial(order, amount_text):
+    """Refund part of an order; amount_text is what the agent typed, such as "19.99"."""
+    cents = to_cents(float(amount_text))
+    if cents > order.paid_cents:
+        raise ValueError("refund exceeds payment")
+    return cents
DIFF
