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
git worktree add -q .worktrees/feat-rst -b feat/rst origin/main
git -C .worktrees/feat-rst mv README.md README.rst
git commit -q --allow-empty -m "Bump"
git push -q origin main
printf "%s\n" .worktrees >> .git/info/exclude
