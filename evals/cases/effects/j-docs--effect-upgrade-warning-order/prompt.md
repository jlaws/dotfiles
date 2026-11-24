---
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "in procedural docs, state a step's condition before its action and a warning's instruction before its risk; .claude/references/documentation/technical-writing-for-devtools.md; hard because the usual warning callout names the danger first, and the draft puts both conditions after their commands"
---

/kb:j-docs turn the draft in ./docs/upgrade.md into a how-to guide that an on-call engineer can follow under pressure. Rewrite the file in place as numbered steps, keep every command, condition, and warning from the draft, and word each step and the warning for that reader.
