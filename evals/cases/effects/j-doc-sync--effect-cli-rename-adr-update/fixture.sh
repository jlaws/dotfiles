set -euo pipefail
cp -R "$1/docs-drift/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
mkdir -p docs/adr/cli
cat > docs/adr/cli/output-flag-name.md <<'MDEOF'
---
status: accepted
topic: cli
created: 2026-09-01
updated: 2026-09-01
---
# Output flag name

## Context
The tool writes JSON lines and needs a flag that names the destination file.

## Decision Drivers
* **Must have** a short flag name for scripts

## Decision
We will name the output flag **--out**.

## Rationale
It is short and easy to type in shell one-liners.

## Consequences
**Gained**: short invocations.
**Accepted**: it differs from the long `--input` naming style.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| none yet | n/a | 2026-09-01 |

## Enforcement
- Owner: `tool.py`
- Pinned by: nothing, convention only

## Reversal Conditions
Revisit if the flag naming style changes to match `--input`.

## Related
- [README](../../../README.md)
MDEOF
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
