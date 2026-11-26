set -euo pipefail
cp -R "$1/py-calc/." .
mkdir -p tools
cat > tools/check_style.py <<'PY'
"""Fail when a line under calc/ is longer than 79 characters."""

import pathlib
import sys

LIMIT = 79
problems = []
for path in sorted(pathlib.Path("calc").rglob("*.py")):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if len(line) > LIMIT:
            problems.append(f"{path}:{number}: line too long ({len(line)} > {LIMIT})")
if problems:
    print("\n".join(problems))
    sys.exit(1)
print("style: clean")
PY
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
