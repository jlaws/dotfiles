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
git switch -q -c feat/median
cat >> calc/ops.py <<'PY'


def median(xs):
    """Return the median of xs. Raises ValueError on an empty list."""
    if not xs:
        raise ValueError("median of an empty list")
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
PY
git commit -q -am "Add median"
cat > tests/test_median.py <<'PY'
import unittest

from calc.ops import median


class MedianTests(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_empty(self):
        with self.assertRaises(ValueError):
            median([])


if __name__ == "__main__":
    unittest.main()
PY
printf '\n`median(xs)` returns the middle value and raises ValueError on an empty list.\n' >> README.md
