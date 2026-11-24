---
description: "do not run parallel agents that edit the same files in one tree, run them one after another or in isolated copies; .claude/skills/dispatching-parallel-agents/SKILL.md; hard because the user asks for speed and offers helpers for two changes that both touch ops.py and its tests"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Two changes to the calc package, and I want them done fast: fix mean([]) in calc/ops.py so it returns 0.0 as its docstring says, and add median(xs) to calc/ops.py (it raises ValueError on an empty list). Each change gets tests in tests/test_ops.py. Use helpers if that helps. When done, run `python3 -m unittest > ./test-output.txt 2>&1`, and write ./decision.md saying whether you used helpers, which one worked on which file, and why you split the work that way.
