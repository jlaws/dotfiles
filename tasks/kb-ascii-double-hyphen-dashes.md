# Task: Replace prose ASCII double-hyphen dashes in the KB trees

## Context
Found in the diff review of `chore/kb-em-dash-sweep` (2026-11-23). That PR removed every em dash
(U+2014) from `.claude/`, `.agents/`, `.codex/`, and `.gemini/`, and `tests/test_agent_config.py`
now holds that. The rule in `CLAUDE.md` (use a comma, semicolon, colon, or parentheses) also covers
dashes typed as ` -- `, which the sweep did not touch.

## Problem
- `git grep -c -E "[a-z] -- [a-z]" -- .claude .agents .codex .gemini` counted about 740 lines on
  that date. Some are CLI flags (`git diff -- path`); many are prose dashes, for example
  `.claude/commands/j-diff-review.md:20` ("not a finding -- except on a security control") and
  `.claude/commands/j-paper-analysis.md:161`.
- The Gemini command tree (`.gemini/antigravity-cli/skills/j-*`) uses ` -- ` in headings and prose
  (`## Phase 1 -- Scope` in `j-defrag`), so its copy of each command now differs in punctuation from
  the other three, which the sweep moved to `:`.

## Scope
1. Classify each ` -- ` hit as a CLI argument separator (keep) or a prose dash (replace by the
   sweep policy: label colon, `;` for a clause, `:` otherwise, parentheses for an aside).
2. Extend the guard in `tests/test_agent_config.py` to reject a prose ` -- ` outside code spans and
   fences, with a control test.
3. Keep the command copies aligned across the four trees.
