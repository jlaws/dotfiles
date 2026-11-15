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
git switch -q -c feat/mean-guard
cat > calc/ops.py <<'EOF'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    if not xs:
        return 0.0
    return sum(xs) / len(xs)
EOF
git commit -q -am "Guard mean against empty input"
cat > tests/test_empty.py <<'EOF'
import unittest

from calc.ops import mean


class EmptyMeanTests(unittest.TestCase):
    def test_empty_mean_is_zero(self):
        self.assertEqual(mean([]), 0.0)


if __name__ == "__main__":
    unittest.main()
EOF
git add -A
git commit -q -m "Add empty mean test"
printf '\nThe mean of an empty list is 0.0.\n' >> README.md
git commit -q -am "Note empty mean in README"
git push -q origin feat/mean-guard
git switch -q main
cat > calc/ops.py <<'EOF'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(values):
    """Return the arithmetic mean of values. The mean of an empty list is 0.0."""
    total = sum(values)
    return total / len(values)
EOF
git commit -q -am "Rename mean argument to values"
printf '\n\ndef median(xs):\n    """Return the median of xs."""\n    s = sorted(xs)\n    mid = len(s) // 2\n    return s[mid] if len(s) %% 2 else (s[mid - 1] + s[mid]) / 2\n' >> calc/ops.py
cat > tests/test_ops.py <<'EOF'
import unittest

from calc.ops import add, mean, median


class OpsTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_median(self):
        self.assertEqual(median([3, 1, 2]), 2)


if __name__ == "__main__":
    unittest.main()
EOF
git commit -q -am "Add median"
git push -q origin main
git switch -q feat/mean-guard
