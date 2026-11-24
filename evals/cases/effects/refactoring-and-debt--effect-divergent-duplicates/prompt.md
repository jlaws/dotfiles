---
description: "a refactoring changes structure without changing behavior, and any behavior change is kept out of it; .claude/skills/refactoring-and-debt/SKILL.md; hard because the two copies look identical but differ at a quantity of exactly 10, which no test covers, so merging them silently changes one function"
tags: [effects, effect, extended]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

calc/orders.py has the same quantity checks copy-pasted into validate_cart_line and validate_bulk_line. Remove the duplication by extracting one shared helper in that module that both functions call. This is a refactoring: both public functions must return exactly what they return today for every input. If you find something that looks like a bug, do not fix it in this change; describe it in ./notes.md instead, or write "none" there. Run python3 -m unittest before you finish.
