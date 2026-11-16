# Task: Use home-anchored reference paths in Codex and Gemini command skills

## Context
Codex and Gemini command skills cite references as `.agents/references/<dir>/<name>`, a path that
resolves against the user's project, not `~/.agents/`. The consult-command change made these paths an
instruction on the default inline path ("Load the relevant references ...").

## Problem
A cloned, untrusted repository can place a file at `.agents/references/<dir>/<name>.md`, and the
model would read it as trusted KB. The form predates this change (59 paths under `.agents/skills`
and 54 under `.codex/prompts` on main). The Gemini agents already use `~/.agents/references/`.
Not verified by running Codex or Gemini.

## Scope
1. Confirm how Codex and Gemini resolve these paths when the project has a same-named file.
2. If the project copy wins, switch command skills and prompts to `~/.agents/references/` and pin the
   form in `tests/test_agent_config.py`.
