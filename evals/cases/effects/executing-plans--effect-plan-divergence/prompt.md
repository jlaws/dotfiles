---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Carry out the plan below. First save it unchanged to ./plans/plan.md and keep that file up to date as you go: tick off tasks in a Progress checklist that records the commit SHA of each finished phase, log judgment calls in a Decision Log, and record anything that differed from what the plan expected in a Surprises & Discoveries section, quoting the actual terminal output. Leave exactly one commit per phase and finish with `git log --oneline > ./log.txt`.

PLAN
Phase 1: Empty mean
- Task 1: In tests/test_empty.py add test_mean_of_empty_list_is_zero asserting mean([]) == 0.0. Run `python3 -m unittest`. Expect a failure (ZeroDivisionError).
- Task 2: In calc/ops.py make mean return 0.0 for an empty list. Run `python3 -m unittest`. Expect "Ran 4 tests" and OK.
Phase 2: Docs
- Task 3: Add a "Functions" section to README.md listing add and mean, and noting that mean([]) is 0.0. Run `python3 -m unittest`. Expect "Ran 4 tests" and OK.
