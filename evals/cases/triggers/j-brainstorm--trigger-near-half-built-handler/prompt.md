---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "a user halfway through an endpoint asks how to handle a missing dict key; .claude/commands/j-brainstorm.md; near: none"
---

I am already halfway through the CSV export endpoint and have the route wired. The handler throws a KeyError when the optional date filter is missing. Show me how to handle the missing key.
