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
git checkout -q -b feat/mean-fix
printf '\n\ndef total(xs):\n    """Return the sum of xs."""\n    return sum(xs)\n' >> calc/ops.py
cat > tests/test_ops.py <<'EOF'
import unittest

from calc.ops import add, mean, total


class OpsTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_total(self):
        self.assertEqual(total([1, 2, 3]), 6)


if __name__ == "__main__":
    unittest.main()
EOF
git add -A
git commit -q -m "Add total helper"
git push -q -u origin feat/mean-fix
