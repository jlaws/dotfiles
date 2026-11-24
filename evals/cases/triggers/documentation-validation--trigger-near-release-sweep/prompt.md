---
tags: [triggers, trigger-near]
max_turns: 6
timeout_seconds: 120
allowed_tools: [Skill, Read, Glob, Grep]
description: "sweeping every doc across a whole release range after shipping; .claude/skills/documentation-validation/SKILL.md; near: post-ship-doc-sync"
---

We shipped v2.0 last week. Sweep every document across the whole release range from v1.9 to v2.0 and list everything that went stale.
