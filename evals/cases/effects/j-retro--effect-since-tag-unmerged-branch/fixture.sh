set -euo pipefail
cp -R "$1/py-calc/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
at() { export GIT_AUTHOR_DATE="$1T10:00:00+0000" GIT_COMMITTER_DATE="$1T10:00:00+0000"; }
at 2026-10-01
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
add_fn() { printf '\n\ndef %s(a, b):\n    """%s a and b."""\n    return a %s b\n' "$1" "$2" "$3" >> calc/ops.py; }
at 2026-10-20
add_fn subtract Subtract -
git commit -q -am "feat: add subtract"
at 2026-10-22
printf '\nSubtract is available as `calc.ops.subtract`.\n' >> README.md
git commit -q -am "docs: mention subtract in README"
at 2026-10-27
printf '\n# TODO: handle bad input\n' >> calc/ops.py
git commit -q -am "chore: note input handling"
at 2026-10-30
printf '\n# subtract reviewed\n' >> calc/ops.py
git commit -q -am "refactor: tidy subtract"
git tag v1.0.0
at 2026-11-02
add_fn multiply Multiply '*'
git commit -q -am "feat: add multiply"
at 2026-11-03
add_fn divide Divide /
git commit -q -am "feat: add divide"
at 2026-11-04
printf '\n# divide: guard zero\n' >> calc/ops.py
git commit -q -am "fix: guard divide by zero"
git branch feat/median
git switch -q feat/median
at 2026-11-05
printf '\ndef median(xs):\n    """Return the median of xs."""\n    s = sorted(xs)\n    return s[len(s) // 2]\n' >> calc/ops.py
git commit -q -am "feat: add median draft"
at 2026-11-06
printf '\n# median: handle even length\n' >> calc/ops.py
git commit -q -am "feat: median even length"
git switch -q main
at 2026-11-07
printf '\n# rename helpers\n' >> calc/ops.py
git commit -q -am "refactor: rename helpers"
at 2026-11-08
add_fn power Power '**'
git commit -q -am "feat: add power"
at 2026-11-09
git revert --no-edit HEAD >/dev/null
git switch -q -c fix/readme-typos
at 2026-11-10
printf '\nSee the tests directory for examples.\n' >> README.md
git commit -q -am "docs: add examples pointer"
git switch -q main
at 2026-11-11
git merge -q --no-ff fix/readme-typos -m "Merge branch fix/readme-typos"
at 2026-11-12
printf '\nDivide raises on zero.\n' >> README.md
git commit -q -am "docs: document divide"
git branch -q -d fix/readme-typos
unset GIT_AUTHOR_DATE GIT_COMMITTER_DATE
