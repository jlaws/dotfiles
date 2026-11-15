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
git switch -q -c feat/stats
cat > calc/stats.py <<'EOF'
"""Descriptive statistics built on calc.ops."""
from calc.ops import mean


def variance(xs):
    m = mean(xs)
    return mean([(x - m) ** 2 for x in xs])


def median(xs):
    # SIMPLIFIED: sorts a copy on every call; ceiling about 100k items; upgrade: quickselect
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
EOF
git add -A
git commit -q -m "Add variance and median"
