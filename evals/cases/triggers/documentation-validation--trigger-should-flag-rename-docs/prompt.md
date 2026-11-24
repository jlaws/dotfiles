---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user renamed a CLI flag and wants the docs checked before committing; .claude/skills/documentation-validation/SKILL.md; should"
---

I just renamed the --out flag to --output in our CLI. Before I commit, make sure the docs still match what the code does.
