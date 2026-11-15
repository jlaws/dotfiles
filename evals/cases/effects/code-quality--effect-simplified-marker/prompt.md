---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Add calc/stats.py with two functions: median(xs) and top_k(xs, k) returning the k largest values in descending order. Keep it quick: a plain repeated max-scan for top_k is fine for now, and I know it will not scale to big inputs.

Mark that shortcut in the code with the deliberate-simplification marker comment our codebase uses for known-ceiling shortcuts, placed on the shortcut itself. The marker must say what breaks and at what size, and what to switch to instead. Do not leave a bare TODO anywhere. Run python3 -m unittest to confirm the existing tests still pass.
