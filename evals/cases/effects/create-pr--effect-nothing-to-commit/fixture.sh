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
git switch -q -c feat/already-pushed
printf '\ndef double(a):\n    """Return twice a."""\n    return a * 2\n' >> calc/ops.py
git commit -q -am "Add double"
git push -q -u origin feat/already-pushed
