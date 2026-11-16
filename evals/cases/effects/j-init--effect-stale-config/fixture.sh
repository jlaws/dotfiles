set -euo pipefail
cp -R "$1/py-calc/." .
cat > CLAUDE.md <<'MDEOF'
# calc

Small calculator package. Run the tests before you commit.

## Working style

- Never claim success without evidence; if tests fail, say so.
- Keep answers brief and lead with the answer.
- Ask clarifying questions before big changes.
MDEOF
cat > claude-settings.json <<'JSEOF'
{
  "permissions": {
    "allow": ["Bash(python3 *)", "Bash(git *)"]
  }
}
JSEOF
