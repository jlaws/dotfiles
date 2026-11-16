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
git switch -q -c feat/subtract
printf '\n\ndef subtract(a, b):\n    """Return a minus b."""\n    return a - b\n' >> calc/ops.py
cat > tests/test_ops.py <<'EOF'
import unittest

from calc.ops import add, mean, subtract


class OpsTests(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_mean(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)


if __name__ == "__main__":
    unittest.main()
EOF
git commit -q -am "Add subtract"
git push -q origin feat/subtract
