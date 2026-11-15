set -euo pipefail
cp -R "$1/py-calc/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
mkdir shop
printf '"""Shop package."""\n' > shop/__init__.py
cat > shop/text.py <<'EOF'
"""Text helpers."""


def normalize_name(name):
    """Collapse whitespace and title-case a person's name."""
    return " ".join(name.split()).title()
EOF
cat > shop/customers.py <<'EOF'
"""Customer records."""
import re


def _normalize(name):
    return " ".join(name.split()).title()


def _valid_email(email):
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email.lower()) is not None


def make_customer(name, email):
    if not _valid_email(email):
        return None
    return {"name": _normalize(name), "email": email.lower()}
EOF
cat > shop/orders.py <<'EOF'
"""Orders."""
import re


def _normalize(name):
    return " ".join(name.split()).title()


def _valid_email(email):
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email.lower()) is not None


def place_order(name, email, items):
    if not _valid_email(email):
        raise ValueError("invalid email")
    return {"customer": _normalize(name), "email": email.lower(), "items": list(items)}
EOF
cat > shop/invoices.py <<'EOF'
"""Invoices."""
import re


def _normalize(name):
    return " ".join(name.split()).title()


def _valid_email(email):
    return re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email.lower()) is not None


def make_invoice(name, email, total):
    if not _valid_email(email):
        raise ValueError("invalid email")
    return {"billed_to": _normalize(name), "email": email.lower(), "total": total}
EOF
cat > tests/test_shop.py <<'EOF'
import unittest

from shop.customers import make_customer
from shop.invoices import make_invoice
from shop.orders import place_order


class ShopTests(unittest.TestCase):
    def test_customer_name_normalized(self):
        self.assertEqual(make_customer("  ada   LOVELACE ", "A@B.io")["name"], "Ada Lovelace")

    def test_customer_invalid_email_is_none(self):
        self.assertIsNone(make_customer("Ada", "nope"))

    def test_order_invalid_email_raises(self):
        with self.assertRaises(ValueError):
            place_order("Ada", "nope", [1])

    def test_invoice_invalid_email_raises(self):
        with self.assertRaises(ValueError):
            make_invoice("Ada", "nope", 5)

    def test_order_and_invoice_names(self):
        self.assertEqual(place_order("ada  lovelace", "a@b.io", [])["customer"], "Ada Lovelace")
        self.assertEqual(make_invoice("ada  lovelace", "a@b.io", 1)["billed_to"], "Ada Lovelace")


if __name__ == "__main__":
    unittest.main()
EOF
git add -A
git commit -q -m "Add shop package"
