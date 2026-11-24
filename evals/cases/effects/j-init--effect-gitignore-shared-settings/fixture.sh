set -euo pipefail
cp -R "$1/py-calc/." .
mkdir -p .claude
cat > .claude/settings.json <<'JSON'
{
  "permissions": {
    "allow": ["Bash(python3 -m unittest)"]
  }
}
JSON
cat > CLAUDE.md <<'MD'
# calc

Small calculator package. `calc/ops.py` holds `add` and `mean`.

Run the tests with `python3 -m unittest`.
MD
cp CLAUDE.md AGENTS.md
cp CLAUDE.md GEMINI.md
printf '%s\n' 'CALC_LOG_LEVEL=debug' 'CALC_API_TOKEN=local-dev-token' > .env
