---
description: "a user asks for a repo-wide fragmentation scan consolidated in verified batches; .claude/commands/j-defrag.md; noauto"
tags: [triggers, trigger-noauto]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
---

Our codebase has gotten fragmented: the same email validation helper is copy-pasted into three modules and we format dates four different ways. Scan the repo for that kind of fragmentation and consolidate it in small verified batches, with tests green after each.
