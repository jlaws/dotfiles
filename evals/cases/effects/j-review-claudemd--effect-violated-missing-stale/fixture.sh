set -euo pipefail
mkdir -p sessions
cat > CLAUDE.md <<'MDEOF'
# Project rules

## Testing
- Use unittest for all tests. Do not use pytest.
- Run `make lint` before every commit.

## Style
- Add type hints to all public functions.
- Keep functions under 40 lines.
MDEOF
cat > sessions/session-1.md <<'MDEOF'
# Session 1 (2026-11-02)
User: add a slugify helper to utils.py with tests
Claude: Created tests/test_slugify.py using pytest fixtures and @pytest.mark.parametrize, then ran `pytest -q`.
User: we use unittest here, rewrite the tests with unittest.TestCase.
Claude: Rewrote the tests with unittest.TestCase.
User: run make lint
Claude: make: *** No rule to make target 'lint'.  Stop.
User: right, we moved to `just lint` months ago. use that.
Claude: Ran `just lint`, clean.
MDEOF
cat > sessions/session-2.md <<'MDEOF'
# Session 2 (2026-11-05)
User: fix the date parsing bug in parse.py
Claude: Fixed it and committed directly on main.
User: do not commit on main. make a branch like fix/date-parsing and commit there.
Claude: Moved the commit to fix/date-parsing.
User: also rename the variable tmp to cutoff in that function. that one is just my taste.
Claude: Renamed it.
MDEOF
cat > sessions/session-3.md <<'MDEOF'
# Session 3 (2026-11-09)
User: add retry to the http client
Claude: Implemented it, ran the tests, committed on main.
User: again on main. branches only, named type/short-description, then open a PR.
Claude: Created feat/http-retry and pushed it.
User: you skipped the linter, run `just lint` before committing.
Claude: Ran `just lint`, clean, amended the commit.
MDEOF
