---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "mock a class with spec so calls to methods it lacks fail, and return realistic data shapes; .claude/skills/language-testing-patterns/SKILL.md; hard because a bare Mock() passes every test here, so nothing in the run pushes a model toward spec"
---

Create calc/convert.py with a class RateClient (method fetch_rate(currency), which would call a remote service and may be left raising NotImplementedError) and a function convert(client, amount, currency) that returns amount multiplied by the rate from client.fetch_rate(currency), rounded to 2 decimals. If the rate is zero or negative, convert raises ValueError.

Write tests/test_convert.py using unittest and unittest.mock with RateClient mocked out (the real one has no network), covering a normal conversion, rounding, and the invalid-rate error. Run python3 -m unittest until it passes.
