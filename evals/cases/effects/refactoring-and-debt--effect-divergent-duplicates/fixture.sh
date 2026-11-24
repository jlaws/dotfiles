set -euo pipefail
cp -R "$1/py-calc/." .
cat > calc/orders.py <<'EOF'
"""Quantity checks for the shopping cart and the bulk-order form."""


def validate_cart_line(qty):
    """Return an error message for a cart line quantity, or None if it is valid."""
    if not isinstance(qty, int):
        return "quantity must be a whole number"
    if qty < 1:
        return "quantity must be at least 1"
    if qty > 10:
        return "quantity must be at most 10"
    return None


def validate_bulk_line(qty):
    """Return an error message for a bulk-order line quantity, or None if it is valid."""
    if not isinstance(qty, int):
        return "quantity must be a whole number"
    if qty <= 0:
        return "quantity must be at least 1"
    if qty >= 10:
        return "quantity must be at most 10"
    return None
EOF
cat > tests/test_orders.py <<'EOF'
import unittest

from calc.orders import validate_bulk_line, validate_cart_line


class OrderTests(unittest.TestCase):
    def test_cart(self):
        self.assertIsNotNone(validate_cart_line(0))
        self.assertIsNone(validate_cart_line(5))
        self.assertIsNotNone(validate_cart_line(11))
        self.assertIsNotNone(validate_cart_line("3"))

    def test_bulk(self):
        self.assertIsNotNone(validate_bulk_line(0))
        self.assertIsNone(validate_bulk_line(5))
        self.assertIsNotNone(validate_bulk_line(20))


if __name__ == "__main__":
    unittest.main()
EOF
