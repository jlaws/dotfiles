set -euo pipefail
mkdir -p shop/plugins
cat > shop/__init__.py <<'PY'
PY
cat > shop/plugins/__init__.py <<'PY'
PY
cat > shop/util.py <<'PY'
"""Small helpers with no internal imports."""

import settings


def debug_enabled():
    return getattr(settings, "DEBUG", False)


def round_cents(value):
    return int(round(value * 100))
PY
cat > settings.py <<'PY'
DEBUG = False
ACTIVE_PLUGINS = ["discount"]
PY
cat > shop/models.py <<'PY'
from shop.util import round_cents


class LineItem:
    def __init__(self, sku, unit_price, qty):
        self.sku = sku
        self.unit_cents = round_cents(unit_price)
        self.qty = qty
PY
cat > shop/pricing.py <<'PY'
from shop.models import LineItem


def subtotal(items):
    return sum(item.unit_cents * item.qty for item in items)


def make_item(sku, unit_price, qty):
    return LineItem(sku, unit_price, qty)
PY
cat > shop/plugins/discount.py <<'PY'
from shop.pricing import subtotal


def apply(items, total):
    if subtotal(items) > 10000:
        return total - 500
    return total
PY
cat > shop/handlers.py <<'PY'
import importlib

import settings
from shop.pricing import make_item, subtotal


def checkout(rows):
    items = [make_item(r["sku"], r["price"], r["qty"]) for r in rows]
    total = subtotal(items)
    for name in settings.ACTIVE_PLUGINS:
        plugin = importlib.import_module("shop.plugins." + name)
        total = plugin.apply(items, total)
    return total
PY
