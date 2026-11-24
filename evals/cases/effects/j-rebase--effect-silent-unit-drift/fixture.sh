set -euo pipefail
cp -R "$1/py-calc/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
printf "%s\n" .remote.git >> .git/info/exclude
cat > calc/ops.py <<'PYEOF'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    return sum(xs) / len(xs)


def pct(part, whole):
    """Return part as a percentage of whole, from 0.0 to 100.0."""
    return 100.0 * part / whole
PYEOF
cat > tests/test_ops.py <<'PYEOF'
import unittest

from calc.ops import add, mean, pct


class OpsTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_pct(self):
        self.assertEqual(pct(1, 4), 25.0)


if __name__ == "__main__":
    unittest.main()
PYEOF
git commit -q -am "Add pct"
git push -q origin main
git switch -q -c feat/share
cat > calc/report.py <<'PYEOF'
"""One-line report entries."""
from calc.ops import pct


def share_line(label, part, whole):
    """Return 'label: NN.N%', the share of whole that part makes up."""
    return "%s: %.1f%%" % (label, pct(part, whole))
PYEOF
git add -A
git commit -q -m "Add share_line report helper"
printf '\nFor a report line, use calc.report.share_line: share_line("cpu", 1, 4) gives "cpu: 25.0%%".\n' >> README.md
git commit -q -am "Document share_line"
git push -q origin feat/share
git switch -q main
cat > calc/ops.py <<'PYEOF'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    return sum(xs) / len(xs)


def pct(part, whole):
    """Return part as a fraction of whole, from 0.0 to 1.0. Callers scale it for display."""
    return part / whole


def clamp(x, low, high):
    """Return x limited to the range low to high."""
    return max(low, min(x, high))
PYEOF
cat > tests/test_ops.py <<'PYEOF'
import unittest

from calc.ops import add, clamp, mean, pct


class OpsTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_pct(self):
        self.assertEqual(pct(1, 4), 0.25)

    def test_clamp(self):
        self.assertEqual(clamp(5, 0, 3), 3)


if __name__ == "__main__":
    unittest.main()
PYEOF
git commit -q -am "Return pct as a fraction and add clamp"
git push -q origin main
git switch -q feat/share
