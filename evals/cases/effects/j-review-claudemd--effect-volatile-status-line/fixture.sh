set -euo pipefail
mkdir -p sessions
cat > CLAUDE.md <<'MDEOF'
# Project rules

Status: sprint 47, 12 open bugs, last deploy 2026-11-02.

- Run `just test` before every commit.
- API handlers in src/api/ return a Result and never raise; callers match on it.
MDEOF
cat > sessions/session-1.md <<'MDEOF'
# Session 1 (2026-11-18)
User: add pagination to the /orders handler
Claude: Added it in src/api/orders.py; a bad cursor returns Result.err instead of raising. Ran `just test`, 58 passed.
User: good. fyi we are in sprint 49 now, 9 open bugs, and we deployed on 11-16.
Claude: Noted.
MDEOF
cat > sessions/session-2.md <<'MDEOF'
# Session 2 (2026-11-21)
User: fix the rounding bug in totals
Claude: Fixed it and committed.
User: did you run just test first?
Claude: No, I committed without running it. Running it now: 58 passed.
User: that rule is already written down. run it before every commit.
MDEOF
