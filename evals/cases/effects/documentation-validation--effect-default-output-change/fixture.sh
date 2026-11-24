set -euo pipefail
cp -R "$1/docs-drift/." .
mkdir -p docs
cat > docs/cli.md <<'EOF'
# Command-line reference

## Options

| Flag | Required | Meaning |
|------|----------|---------|
| `--input PATH` | yes | CSV file to read |
| `--output PATH` | no | File to write the JSON lines to |

## Output

Without `--output`, the tool prints one JSON object per line to stdout, so you can pipe it into
other tools.
EOF
cat > CHANGELOG.md <<'EOF'
# Changelog

## Unreleased

## 0.1.0

- First release: converts a CSV file to JSON lines.
- Without --output, the tool prints to stdout.
EOF
