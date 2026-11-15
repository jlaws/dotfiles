---
type: llm
focus: { source: file, path: claudemd-proposal.md }
---

All claims are required.
PASS if the proposal flags the unittest rule as violated in session 1 (an enforcement gap, since the rule exists) and flags `make lint` as stale with `just lint` as the replacement.
PASS if the proposal adds a rule against committing on main with branches named type/short-description, citing sessions 2 and 3.
FAIL if the proposal adds a rule about renaming tmp to cutoff, or any other one-off preference.
FAIL if the proposal says it edited CLAUDE.md.
