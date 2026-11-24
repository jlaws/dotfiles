set -euo pipefail
cp -R "$1/py-calc/." .
cat > calc/fmt_money.py <<'EOF'
"""Money formatting."""


def money(amount):
    """Format an amount in dollars with two decimals."""
    return "$%.2f" % amount
EOF
cat > calc/fmt_percent.py <<'EOF'
"""Percent formatting."""


def percent(ratio):
    """Format a ratio as a percentage with one decimal."""
    return "%.1f%%" % (ratio * 100)
EOF
cat > calc/report.py <<'EOF'
"""Plain-text report."""
from calc.fmt_money import money
from calc.fmt_percent import percent
from calc.ops import mean


def report(prices, share):
    return "average price %s, share %s" % (money(mean(prices)), percent(share))
EOF
cat > tests/test_fmt.py <<'EOF'
import unittest

from calc.fmt_money import money
from calc.fmt_percent import percent
from calc.report import report


class FmtTests(unittest.TestCase):
    def test_money(self):
        self.assertEqual(money(3), "$3.00")

    def test_percent(self):
        self.assertEqual(percent(0.125), "12.5%")

    def test_report(self):
        self.assertEqual(report([2, 4], 0.5), "average price $3.00, share 50.0%")


if __name__ == "__main__":
    unittest.main()
EOF
cat > README.md <<'EOF'
# calc

A small calculator package.

## Formatting

```python
from calc.fmt_money import money
from calc.fmt_percent import percent

money(3)        # '$3.00'
percent(0.125)  # '12.5%'
```

Run the tests with:

```bash
python3 -m unittest
```
EOF
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
