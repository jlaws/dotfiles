---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "when a shipped change contradicts a decision record, update it in place: revise the Decision, move the old approach to Ruled Out with reason and date, and bump updated; .claude/skills/post-ship-doc-sync/SKILL.md; hard because a model treats a decision record as history and leaves it asserting the old flag, or appends a note instead of revising the Decision, Ruled Out, and updated fields"
---

Release v0.2 of this tool shipped. Its only user-facing change renamed the --out flag to --output; the code already has it. Today is 2026-11-24. The repo's documentation is README.md and the decision records docs/adr/cli/output-flag-name.md and docs/adr/io/csv-dialect.md, indexed by docs/adr/README.md.

Bring the project's documentation in line with what shipped, including any recorded decision the release overturned, and keep each changed document's own metadata current. Make only the changes the shipped change requires, and do not commit. Save ./sync-summary.md listing every doc file you changed and why.
