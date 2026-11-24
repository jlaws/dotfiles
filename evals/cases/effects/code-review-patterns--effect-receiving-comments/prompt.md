---
description: "if any review comment is unclear, stop and ask a specific question about it before implementing any comment, instead of implementing the clear ones first; .claude/skills/code-review-patterns/SKILL.md; hard because a model tends to fix the clear comment right away and ask about the unclear one later"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A reviewer left these comments on my change to calc/ops.py. Handle them as I would want a careful author to. Save a reply to each, numbered 1 to 4, in ./replies.md, and edit the code only for the comments you decide to act on. In replies.md, also say which comments you have implemented so far.

1. "mean() raises on an empty list but its docstring says 0.0. Please make the code match."
2. "Rename add() to plus() across the repo, it reads better." (Other modules and the tests call add().)
3. "This module needs to be more robust."
4. "Add median(), stdev(), and mode() now, we will probably want them later." (No caller needs them.)
