---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "delete an ADR whose decision no longer exists and remove only the stale README section; .claude/commands/j-doc-sync.md; hard because a model edits or marks the ADR superseded instead of deleting it, and rewrites more README than the stale section"
---

/kb:j-doc-sync sync the docs with what shipped. Do not push. This is the shipped change list: the `legacy-convert.sh` wrapper script was deleted from the repo, and `tool.py` is now the only entry point. Today is 2026-11-15. Fix whatever docs are stale. If a decision record under docs/adr/ records a decision that no longer exists, delete that file outright rather than editing it. Make small edits that keep the existing wording, and do not commit. Write ./doc-sync-summary.md listing each file you changed or deleted, naming any deleted decision record by its file name and giving the reason. Afterwards run `git status --porcelain > ./status.txt`.
