---
description: "an integration test fails one CI run in five and retries have not explained it; .claude/skills/debugging-methodology/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

The checkout integration test passes on my laptop but fails about one run in five on CI with 'expected 3 items, got 2'. I've retried it a dozen times and have no theory. Help me track it down.
