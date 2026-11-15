set -euo pipefail
mkdir -p sessions
cat > CLAUDE.md <<'MDEOF'
# Working agreement

- Be concise. Keep answers short.
- Explain every step in full detail so nothing is unclear.
- Write clean, readable code.
- Use descriptive variable names.
- When the nightly import job fails with error E1042, check that the vendor CSV has a trailing newline before anything else.
- Never delete files without asking.
MDEOF
cat > sessions/session-1.md <<'MDEOF'
# Session 1 (2026-11-03)
User: why is the report generator slow?
Claude: (six paragraphs walking through every function it opened, then the answer)
User: way too long, just tell me the cause and the diff.
Claude: The cause is the N+1 query in build_rows. Diff attached.
User: good. run the full test suite and paste the output.
Claude: Pasted the unittest output, 41 tests OK.
User: that is what I want every time, the pasted output, not "tests pass".
MDEOF
cat > sessions/session-2.md <<'MDEOF'
# Session 2 (2026-11-08)
User: add a --verbose flag to the CLI
Claude: Added it. Then explained all eleven steps of the change in detail.
User: I said be concise. skip the walkthrough.
Claude: Understood, shorter now.
User: run the tests and paste the output
Claude: Pasted the output, 43 tests OK.
User: good, keep doing that.
MDEOF
