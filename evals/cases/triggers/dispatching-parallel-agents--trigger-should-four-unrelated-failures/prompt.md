---
description: "four test files fail in unrelated modules and the user wants them investigated concurrently; .claude/skills/dispatching-parallel-agents/SKILL.md; should"
tags: [triggers, trigger-should]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Four test files are failing in unrelated modules: auth, billing, search, and exports. Investigate them concurrently, one helper per module, and report the root cause of each.
