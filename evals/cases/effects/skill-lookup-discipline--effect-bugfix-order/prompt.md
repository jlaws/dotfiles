---
description: "for a bug, the debugging process skill comes before any code-changing skill, and an instruction saying what to do does not waive how; .claude/skills/skill-lookup-discipline/SKILL.md; hard because a one-line just-fix-it request invites skipping the skill check"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Just fix the bug in calc/ops.py: mean([]) should return 0.0 and tests/test_empty.py already covers it. It is a one-line change. Before touching anything, write ./skills-used.md listing, in the order you will apply them, each existing skill from this knowledge base that applies, with one line on why, and include a line saying whether the "just fix it" wording changes which skills apply. Include only the skills the task needs. Then do the fix and run python3 -m unittest.
