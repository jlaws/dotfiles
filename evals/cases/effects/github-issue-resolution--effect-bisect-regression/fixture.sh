set -euo pipefail
cp -R "$1/py-calc/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
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
git add -A
git commit -q -m "Return 0.0 for the mean of an empty list (#12)"
git tag v0.2
printf '%s\n' '' 'Released as v0.2.' >> README.md
git add -A
git commit -q -m "Note the v0.2 release in README (#15)"
cat > calc/ops.py <<'EOF'
"""Arithmetic helpers."""


def add(a, b):
    """Return the sum of a and b."""
    return a + b


def mean(xs):
    """Return the arithmetic mean of xs. The mean of an empty list is 0.0."""
    return sum(xs)/len(xs)
EOF
git add -A
git commit -q -m "Tidy ops helpers (#20)"
printf '%s\n' '' 'Issues and pull requests are welcome.' >> README.md
git add -A
git commit -q -m "Add contributing note (#21)"
cp "$1/py-calc/calc/ops.py" calc/ops.py
git add -A
git commit -q -m "Format ops.py (#23)"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main --tags
printf "%s\n" .remote.git >> .git/info/exclude
