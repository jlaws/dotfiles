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
git checkout -q -b feat/readme-rst
git mv README.md README.rst
git commit -q -m "Rename README to rst"
git checkout -q main
git commit -q --allow-empty -m "Bump"
git push -q origin main
git reset -q --hard HEAD~1
git checkout -q feat/readme-rst
