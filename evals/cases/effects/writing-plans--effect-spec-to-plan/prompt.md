---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

I want to add a median(xs) function to calc/ops.py (it raises ValueError on an empty list), a command line entry point so that `python3 -m calc median 3 1 2` prints 2, and a README section for both. Plan the work before any code is written and save the plan to ./plans/plan.md. Write no code and do not commit the plan.

The plan must: open with one sentence stating what changes for a user; start its first phase by fetching and branching off origin/main, then write the README before any test or code task; use test-first phases for the code; end with a validation phase that names the gate command and what it must print, or says N/A with a reason; state how many pull requests the work ships as, which phases go in each, and the estimated changed lines per PR; and give function signatures and test assertions rather than full function bodies.
