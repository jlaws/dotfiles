---
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user dropping a Python version wants docs and badges checked before opening the PR; .claude/skills/documentation-validation/SKILL.md; should"
---

I'm about to open a PR that drops Python 3.8 support in pyproject.toml. Make sure nothing in the docs or badges still claims we support it.
