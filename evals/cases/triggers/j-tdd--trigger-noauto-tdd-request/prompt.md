---
description: "a user asks for a test-first fix with a watched failing test and a commit; .claude/commands/j-tdd.md; noauto"
tags: [triggers, trigger-noauto]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Add input validation to a parse_date helper: first write a test that shows a malformed date slipping through and watch it fail, then make it pass, then commit.
