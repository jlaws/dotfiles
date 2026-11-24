---
description: "assess before coding: measured scope, the driving complexity factor, one strategy, and unresolved behavior questions written before the port; .claude/skills/code-migration/SKILL.md; hard because the port is small enough that a model writes the code first and silently picks one behavior for mean([])"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Port the calc package (calc/ops.py) to JavaScript as an ES module at ./js/ops.js, with the same function names.

Before porting, save a migration assessment to ./migration-assessment.md: the scope in numbers you measured from the repo, the overall complexity with the factor that drives it, the single strategy you chose and why, and a list of unresolved questions about behavior you are not sure the port should preserve. Write comparison tests for the Python code first, run them to confirm they pass on the old code, then write the JS tests in ./js/ops.test.js (node --test) asserting the same behavior and run them against the port.
