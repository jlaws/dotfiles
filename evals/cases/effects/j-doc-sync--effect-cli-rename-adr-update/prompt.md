---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "sync docs with only what shipped, and update a contradicted ADR in place with a dated Ruled Out row and a bumped updated date; .claude/commands/j-doc-sync.md; hard because the change list also names a planned flag a model may document, and the ADR upkeep follows a house format"
---

/kb:j-doc-sync sync the docs with what shipped. Do not push. This is the shipped change list: (1) tool.py renamed the output flag from `--out` to `--output`, still defaulting to stdout when omitted; (2) a `--format` flag is planned for the next release but has NOT shipped. Today is 2026-11-15. Fix whatever docs are stale, including the decision record at docs/adr/cli/output-flag-name.md that the rename contradicts: revise it in place, record the old approach as ruled out with its reason and today's date, and bump its updated date. Make small edits that keep the existing wording, document nothing unshipped, and do not commit. Write ./doc-sync-summary.md listing each file you changed and each item you deliberately left undocumented, with the reason.
