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
git checkout -q -b feat/total
cat > calc/stats.py <<'EOF'
"""Aggregate helpers built on calc.ops."""

from calc.ops import add


def total(xs):
    """Return the sum of xs, adding one value at a time."""
    result = 0
    for x in xs:
        result = add(result, x)
    return result
EOF
cat > tests/test_stats.py <<'EOF'
import unittest

from calc.stats import total


class StatsTests(unittest.TestCase):
    def test_total_sums_values(self):
        self.assertEqual(total([1, 2, 3]), 6)

    def test_total_of_empty_list_is_zero(self):
        self.assertEqual(total([]), 0)

    @unittest.skip("weighted totals are planned for 0.3")
    def test_weighted_total(self):
        self.assertEqual(total([1, 2], weights=[1, 1]), 3)


if __name__ == "__main__":
    unittest.main()
EOF
git add -A
git commit -q -m "Add total"
git checkout -q main
cat > calc/ops.py <<'EOF'
"""Arithmetic helpers."""


def plus(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    return sum(xs) / len(xs)
EOF
cat > tests/test_ops.py <<'EOF'
import unittest

from calc.ops import mean, plus


class OpsTests(unittest.TestCase):
    def test_plus(self):
        self.assertEqual(plus(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
EOF
git add -A
git commit -q -m "Rename add to plus"
git push -q origin main
git checkout -q feat/total
