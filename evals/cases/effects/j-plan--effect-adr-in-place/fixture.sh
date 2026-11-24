set -euo pipefail
cp -R "$1/py-calc/." .
mkdir -p docs/adr/calc docs/adr/testing
cat > docs/adr/calc/empty-input.md <<'MDEOF'
---
status: accepted
topic: calc
created: 2026-06-02
updated: 2026-06-02
---
# Mean of an empty list

## Context
Callers average lists that can be empty, such as a day with no samples.

## Decision
`mean([])` returns `0.0`.

## Rationale
A dashboard that averages an empty day shows 0.0 instead of crashing.

## Consequences
**Gained**: callers need no empty check.
**Accepted**: an empty input looks the same as a true zero mean.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Let ZeroDivisionError escape | leaks an implementation detail to callers | 2026-06-02 |

## Reversal Conditions
Revisit if callers need to tell an empty input from a true zero mean.
MDEOF
cat > docs/adr/testing/test-runner.md <<'MDEOF'
---
status: accepted
topic: testing
created: 2026-05-10
updated: 2026-05-10
---
# Test runner

## Context
The package has no third-party dependencies.

## Decision
Tests use the standard library's unittest, run with `python3 -m unittest`.

## Rationale
No install step before the suite runs.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| pytest | adds a dependency | 2026-05-10 |

## Reversal Conditions
Revisit if the package takes on a test dependency for another reason.
MDEOF
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
printf "%s\n" plans/ >> .git/info/exclude
