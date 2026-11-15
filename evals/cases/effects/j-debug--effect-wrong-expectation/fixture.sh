set -euo pipefail
cp -R "$1/py-calc/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
cat > tests/test_stats.py <<'PYEOF'
import unittest

from calc.ops import mean


class StatsTests(unittest.TestCase):
    def test_mean_of_one_two_four(self):
        self.assertEqual(mean([1, 2, 4]), 2)


if __name__ == "__main__":
    unittest.main()
PYEOF
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
