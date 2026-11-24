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
cat > calc/ops.py <<'PY'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def median(xs):
    """Return the median of xs."""
    ordered = sorted(xs)
    return ordered[len(ordered) // 2]
PY
cat > tests/test_median.py <<'PY'
import unittest

from calc.ops import median


class MedianTests(unittest.TestCase):
    def test_odd_length(self):
        self.assertEqual(median([3, 1, 2]), 2)


if __name__ == "__main__":
    unittest.main()
PY
