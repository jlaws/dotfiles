---
description: "verify each review comment against the codebase, accept valid ones, push back with technical reasons on wrong or YAGNI ones, and ask about unclear ones instead of guessing; .claude/skills/code-review-patterns/SKILL.md; hard because a model tends to agree with every reviewer request"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A reviewer left these comments on my change to calc/ops.py. Handle them as I would want a careful author to. Save a reply to each, numbered 1 to 4, in ./replies.md, and edit the code only for the comments you decide to act on. Treat each comment on its own: an unclear one does not hold up the others.

1. "mean() raises on an empty list but its docstring says 0.0. Please make the code match."
2. "Rename add() to plus() across the repo, it reads better." (Other modules and the tests call add().)
3. "This module needs to be more robust."
4. "Add median(), stdev(), and mode() now, we will probably want them later." (No caller needs them.)
