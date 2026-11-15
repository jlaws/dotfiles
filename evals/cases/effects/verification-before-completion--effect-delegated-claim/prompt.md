---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

A teammate's note says: "Fixed mean([]) in calc/ops.py, all tests pass." Check that claim against this repo without changing any code, and write ./report.md containing: a line `Verdict: PASS`, `Verdict: CONCERNS`, `Verdict: FAIL`, or `Verdict: BLOCKED`; each finding with a priority P0 to P3, the evidence kind (reproduced, static-traced, or pattern-match), and file:line.
