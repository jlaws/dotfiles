---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user changed a default port and wants README and docs checked before pushing; .claude/skills/documentation-validation/SKILL.md; should"
---

I just changed the server's default port from 8080 to 9000 in config.py. Before I push, check that README and the docs folder don't still say 8080.
