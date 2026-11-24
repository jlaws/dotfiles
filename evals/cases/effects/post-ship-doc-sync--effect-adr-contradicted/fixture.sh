set -euo pipefail
cp -R "$1/docs-drift/." .
mkdir -p docs/adr/cli docs/adr/io
cat > docs/adr/README.md <<'EOF'
# Decision records

One file per decision, grouped by topic. Every file here is currently true: a changed decision is
edited in place and a dead one is deleted.
EOF
cat > docs/adr/cli/output-flag-name.md <<'EOF'
---
status: accepted
topic: cli
created: 2026-03-02
updated: 2026-03-02
---
# Name of the output flag

## Context
The tool writes JSON lines either to stdout or to a file the user names.

## Decision
The flag that names the output file is `--out`.

## Rationale
It is short to type and reads naturally next to `--input`.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| `-o` only | Single-letter flags are hard to search for in scripts | 2026-03-02 |

## Reversal Conditions
Users report confusion between `--out` and other tools' flags.
EOF
cat > docs/adr/io/csv-dialect.md <<'EOF'
---
status: accepted
topic: io
created: 2026-03-02
updated: 2026-03-02
---
# Input dialect

## Context
Input files come from spreadsheet exports.

## Decision
`--input` reads comma-delimited CSV with a header row. Other delimiters are not supported.

## Rationale
Every current export uses commas; one dialect keeps the reader simple.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Auto-detecting the delimiter | Guesses wrong on single-column files | 2026-03-02 |

## Reversal Conditions
A supported export starts producing tab- or semicolon-delimited files.
EOF
