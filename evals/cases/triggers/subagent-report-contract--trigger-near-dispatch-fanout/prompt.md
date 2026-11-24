---
description: "a user wants independent reviews split across parallel reviewers, the caller side of a dispatch; .claude/skills/subagent-report-contract/SKILL.md; near: dispatching-parallel-agents"
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

I have three independent modules to review: auth, billing, and search. Set up the work so three reviewers can run at the same time and tell me how you would split it.
