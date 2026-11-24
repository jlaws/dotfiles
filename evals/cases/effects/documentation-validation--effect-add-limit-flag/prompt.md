---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "before calling a feature change done, check the docs its change type points to against the code, fix stale ones in the same change, and state the Docs decision; .claude/skills/documentation-validation/SKILL.md; hard because the stale --out flag in the README predates the change, so a model that documents only the new flag leaves a broken usage line"
---

Add a --limit N option to tool.py that stops after N rows. Finish the change the way you would before committing it. README.md must document --limit with a short description, and every flag its usage example shows must be one tool.py accepts. Then write ./change-notes.md with one line starting `Docs:` that either says which docs you updated or says N/A with a one-line reason, plus a line `Checked:` listing each doc file you opened.
