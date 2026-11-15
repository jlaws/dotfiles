---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A reviewer left these comments on my change to calc/ops.py. Handle them as I would want a careful author to. Save a reply to each, numbered 1 to 4, in ./replies.md, and edit the code only for the comments you decide to act on.

1. "mean() raises on an empty list but its docstring says 0.0. Please make the code match."
2. "Rename add() to plus() across the repo, it reads better." (Other modules and the tests call add().)
3. "This module needs to be more robust."
4. "Add median(), stdev(), and mode() now, we will probably want them later." (No caller needs them.)
