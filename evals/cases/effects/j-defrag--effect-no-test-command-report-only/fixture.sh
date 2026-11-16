set -euo pipefail
cp -R "$1/py-calc/." .
rm tests/test_ops.py tests/__init__.py
rmdir tests
printf '# calc\n\nA small calculator package.\n' > README.md
cat > calc/report.py <<'EOF'
"""Plain-text report."""


def _avg(xs):
    return sum(xs) / len(xs)


def report(xs):
    return "average: %s" % _avg(xs)
EOF
cat > calc/summary.py <<'EOF'
"""One-line summary."""


def _avg(xs):
    return sum(xs) / len(xs)


def summarize(xs):
    return "n=%d avg=%s" % (len(xs), _avg(xs))
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
