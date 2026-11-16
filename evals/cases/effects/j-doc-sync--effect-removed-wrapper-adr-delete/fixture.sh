set -euo pipefail
cp -R "$1/docs-drift/." .
git init -q -b main
git config user.name kbeval
git config user.email kbeval@example.invalid
cat > README.md <<'MDEOF'
# tool

Convert a CSV file to JSON lines.

## Usage

```bash
python3 tool.py --input data.csv --output data.jsonl
```

`--input` is required. `--output` names the file to write; without it the tool prints to stdout.

## Legacy wrapper

Shell users can call `./legacy-convert.sh data.csv`, which runs `tool.py` with the same arguments.
MDEOF
mkdir -p docs/adr/cli
cat > docs/adr/cli/wrapper-entrypoint.md <<'MDEOF'
---
status: accepted
topic: cli
created: 2026-08-01
updated: 2026-08-01
---
# Ship a shell wrapper as an entry point

## Context
Some users run the tool from shell scripts and want a command without the `python3` prefix.

## Decision Drivers
* **Should support** shell users

## Decision
We will ship **legacy-convert.sh** as a second entry point next to `tool.py`.

## Rationale
It costs one small script.

## Consequences
**Gained**: shell users avoid the interpreter prefix.
**Accepted**: two entry points to keep in sync.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| none yet | n/a | 2026-08-01 |

## Enforcement
- Owner: `legacy-convert.sh`
- Pinned by: nothing, convention only

## Reversal Conditions
Revisit if the wrapper is removed.

## Related
- [README](../../../README.md)
MDEOF
git add -A
git commit -q -m "Initial commit"
git init -q --bare .remote.git
git remote add origin ./.remote.git
git push -q origin main
printf "%s\n" .remote.git >> .git/info/exclude
