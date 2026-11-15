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
git switch -q -c feat/summarize
cat > calc/summary.py <<'EOF'
"""One-line summaries of numeric data."""
from calc.ops import mean


def summarize(xs):
    return "mean=%s" % mean(xs)
EOF
git add -A
git commit -q -m "Add summarize helper"
printf '\nUse calc.summary.summarize for a one-line summary.\n' >> README.md
git commit -q -am "Document summarize"
git push -q origin feat/summarize
git switch -q main
sed -i.bak 's/def mean(xs)/def average(xs)/' calc/ops.py
rm calc/ops.py.bak
sed -i.bak 's/mean/average/g' tests/test_ops.py
rm tests/test_ops.py.bak
git commit -q -am "Rename mean to average"
git push -q origin main
git switch -q feat/summarize
